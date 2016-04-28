-- Track accuracy
opt.lastAcc = opt.lastAcc or 0
opt.bestAcc = opt.bestAcc or 0
-- We save snapshots of the best model only when evaluating on the full validation set
trackBest = (opt.validIters * opt.validBatch == ref.valid.nsamples)

-- The dimensions of 'final predictions' are defined by the opt.task file
-- This allows some flexibility for post-processing of the network output
preds = torch.Tensor(ref.valid.nsamples, unpack(predDim))

-- We also save the raw output of the network (in this case heatmaps)
if type(outputDim[1]) == "table" then predHMs = torch.Tensor(ref.valid.nsamples, unpack(outputDim[#outputDim]))
else predHMs = torch.Tensor(ref.valid.nsamples, unpack(outputDim)) end


if(opt.useSPEN) then
    criterion = nn.MSECriterion()
    criterion:cuda()
end


batchers = {}

paths.dofile('batcher.lua')

batchers['train'] = opt.dataCache ~= "" and Batcher(opt.dataCache,opt.trainBatch)
batchers['predict'] = opt.validDataCache ~= "" and Batcher(opt.validDataCache,opt.validBatch)


-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local r = ref[tag]

    if tag == 'train' then
        print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
        model:training()
        set = 'train'
        isTesting = false -- Global flag
    else
        if tag == 'predict' then print("==> Generating predictions...") end
        model:evaluate()
        set = 'valid'
        isTesting = true
    end
    local currAvg


    local batcher = batchers[tag]

    --xlua.progress(0, r.iters)
    local numProcessed = 0
    local blockCount = 0
    for i=1,r.iters do
        collectgarbage()

        local output,err,idx
       -- xlua.progress(i, r.iters)

        -- Load in data
        if tag == 'predict' or (tag == 'valid' and trackBest) then idx = i end

        local input, label
        ---todo: for debugging


        if(batcher) then
            input, label, endfile = batcher:getData()
            if(endfile and (tag == "predict" or tag == "valid")) then break end
        else
            input, label = loadData(set, idx, r.batchsize)
        end
        blockCount = blockCount + 1
        -- Do a forward pass and calculate loss
        prebatch()
        output = model:forward(input)
        if(opt.useSPEN and (type(label) == "table")) then
            label = label[#label]
        end
        if(opt.useSPEN and (type(output) == "table")) then
            output = output[#output]
        end

        err = criterion:forward(output, label)

        -- Training: Do backpropagation and optimization
        if tag == 'train' then
            model:zeroGradParameters()
            local dfdo = criterion:backward(output, label)
            model:backward(input, dfdo)
            local function evalFn(x) return err, gradparam end
            optfn(evalFn, param, optimState)

        -- Validation: Get flipped output
        else
            output = applyFn(function (x) return x:clone() end, output)
            local flip_ = customFlip or flip
            local shuffleLR_ = customShuffleLR or shuffleLR
            local flippedOut = model:forward(flip_(input))
            flippedOut = applyFn(function (x) return flip_(shuffleLR_(x)) end, flippedOut)
            output = applyFn(function (x,y) return x:add(y):div(2) end, output, flippedOut)

        end

        -- Synchronize with GPU
        if opt.GPU ~= -1 then cutorch.synchronize() end

        -- If we're generating predictions, save output
        if(opt.useSPEN) then
            output = {output}
            label = {label}
        end

        if tag == 'predict' or (tag == 'valid' and trackBest) then
            local oo = type(outputDim[1]) == "table" and output[#output] or output
            predHMs:narrow(1,numProcessed+1,oo:size(1)):copy(oo)
            if postprocess then preds:sub(numProcessed+1,numProcessed + oo:size(1)):copy(postprocess(set,idx,{oo})) end
        end

        -- Calculate accuracy
        numProcessed = numProcessed + input:size(1)

        local acc = accuracy(output, label)
        avgLoss = avgLoss + err
        avgAcc = avgAcc + acc
        local gamma = 0.98
        currAvg = (i == 1) and err or (gamma*currAvg + (1 - gamma)*err)
        if(i % 5 == 0) then print(tag..' loss-'..i..': '..currAvg.." "..err) end
        --xlua.progress(i,r.iters)
    end
    --todo: remove
    classifier:clearState()
    torch.save('classifier.t7',classifier)

    if(tag == "predict" or tag == "valid") then assert(numProcessed == r.iters,"numProcessed = "..numProcessed.." r.iters = "..r.iters) end
    avgLoss = avgLoss / blockCount
    avgAcc = avgAcc / blockCount

    local epochStep = torch.floor(ref.train.nsamples / (r.iters * r.batchsize * 2))
    if tag == 'train' and epoch % epochStep == 0 then
        if avgAcc - opt.lastAcc < opt.threshold then
            isFinished = true --Training has plateaued
        end
        opt.lastAcc = avgAcc
    end

    -- Print and log some useful performance metrics
    print(string.format("      %s : Loss: %.7f Acc: %.4f"  % {set, avgLoss, avgAcc}))
    if r.log then
        r.log:add{
            ['epoch     '] = string.format("%d" % epoch),
            ['loss      '] = string.format("%.6f" % avgLoss),
            ['acc       '] = string.format("%.4f" % avgAcc),
            ['LR        '] = string.format("%g" % optimState.learningRate)
        }
    end

    if tag == 'train' and opt.snapshot ~= 0 and epoch % opt.snapshot == 0 then
        -- Take an intermediate training snapshot
        model:clearState()
        local modelPath = paths.concat(opt.save, 'model_' .. epoch .. '.t7')
        print('saving to '..modelPath)
        torch.save(modelPath, model)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
    elseif tag == 'valid' and trackBest and avgAcc > opt.bestAcc then
        -- A new record validation accuracy has been hit, save the model and predictions
        predFile = hdf5.open(opt.save .. '/best_preds.h5', 'w')
        predFile:write('heatmaps', predHMs)
        if postprocess then predFile:write('preds', preds) end
        predFile:close()
        model:clearState()
        torch.save(paths.concat(opt.save, 'best_model.t7'), model)
        torch.save(paths.concat(opt.save, 'optimState.t7'), optimState)
        opt.bestAcc = avgAcc
    elseif tag == 'predict' then
        -- Save final predictions
        predFile = hdf5.open(opt.save .. '/preds.h5', 'w')
        predFile:write('heatmaps', predHMs)
        if postprocess then predFile:write('preds', preds) end
        predFile:close()
    end
end

function train() step('train') end
function valid() step('valid') end
function predict() step('predict') end
