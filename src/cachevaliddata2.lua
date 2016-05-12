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


batchers = {}

paths.dofile('batcher.lua')

batchers['train'] = opt.dataCache ~= "" and Batcher(opt.dataCache,opt.trainBatch,false)
batchers['predict'] = opt.validDataCache ~= "" and Batcher(opt.validDataCache,opt.validBatch,true)

-- Main processing step
function step(tag)
    local batcher = batchers[tag]

    local avgLoss, avgAcc = 0.0, 0.0
    local r = ref[tag]
    local idx

    if tag == 'train' then
        print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
        set = 'train'
        isTesting = false -- Global flag
    else
        if tag == 'predict' then print("==> Generating predictions...") end
        set = 'valid'
        isTesting = true
    end
    local currAvg

    if tag == 'predict' or (tag == 'valid' and trackBest) then idx = 1 end
    local nextInput, nextLabel
    if(not batcher) then
        nextInput, nextLabel = loadData(set, idx, r.batchsize)
    else
        nextInput, nextLabel = batcher:getData()
    end

    local blockCount = 0
    local numProcessed = 0


    local numBlocks = 1
    local numPerFile = 2958
    local numProcessed = 0
    local currAvg
    for i=1,numBlocks do
        collectgarbage()
        local j = 0
        while(j < numPerFile) do
            xlua.progress(j,numPerFile)
            local input, label = nextInput, nextLabel
            blockCount = blockCount + 1

            local si = input:size()
            numProcessed = numProcessed + si[1]
            inputs = inputs or torch.Tensor(numPerFile,si[2],si[3],si[4])
            print((j+1).." "..si[1])
            inputs:narrow(1,j+1,si[1]):copy(input)

            local li = label[#label]:size()
            labels = labels or torch.Tensor(numPerFile,li[2],li[3],li[4])
            labels:narrow(1,j+1,si[1]):copy(label[#label])
            j = j+si[1]
            -- Load up next sample, runs simultaneously with GPU
            -- If idx is nil, loadData will choose a sample at random
            if tag == 'predict' or (tag == 'valid' and trackBest) then idx = blockCount+1 end
            if j < numPerFile then 
                if(not batcher) then
                    local numToTake = (j + r.batchsize <= numPerFile) r.batchsize or (numPerFile - j)
                    nextInput, nextLabel = loadData(set, idx, numToTake)
                else
                    nextInput, nextLabel = batcher:getData()
                end
            end
        end
        assert(numProcessed == numPerFile,"num processed = "..numProcessed.." numPerFile = "..numPerFile)
        local ofile = opt.save.."/"..tag.."-data-"..i..".t7"
        print(ofile)
        torch.save(ofile,{inputs,labels})

    end

            local ofile = opt.save.."/"..tag.."-data-"..i..".t7"
        print(ofile)
        torch.save(ofile,{inputs,labels})


    if(tag == "predict" or tag == "valid") then 
        assert(numProcessed == r.iters,"numProcessed = "..numProcessed.." r.iters = "..r.iters) 
        batchers['predict'] = opt.validDataCache ~= "" and Batcher(opt.validDataCache,opt.validBatch,true)
    end
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
