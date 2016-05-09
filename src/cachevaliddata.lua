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



-- Main processing step
function step(tag)
    local avgLoss, avgAcc = 0.0, 0.0
    local r = ref[tag]

    if tag == 'train' then
        print("==> Starting epoch: " .. epoch .. "/" .. (opt.nEpochs + opt.epochNumber - 1))
        set = 'train'
        isTesting = false -- Global flag
    else
        if tag == 'predict' then print("==> Generating predictions...") end
        set = 'valid'
        isTesting = true
    end
    local numBlocks = 1
    local numPerFile = 2958
    local currAvg
    for i=1,numBlocks do
        collectgarbage()
        local inputs,labels
        local j = 0
        local numMinibatches = 1
        while(j < numPerFile) do
            xlua.progress(j, numPerFile)

            -- Load in data
            --if tag == 'predict' or (tag == 'valid' and trackBest) then idx = i end
            local input, label = loadData(set, numMinibatches, r.batchsize)
            local si = input:size()
            inputs = inputs or torch.Tensor(numPerFile,si[2],si[3],si[4])
            inputs:narrow(1,j+1,si[1]):copy(input)

            local li = label[2]:size()
            labels = labels or torch.Tensor(numPerFile,li[2],li[3],li[4])
            labels:narrow(1,j+1,si[1]):copy(label[2])
            j = j+si[1]
            numMinibatches = numMinibatches + 1
        end
        local ofile = opt.save.."/"..tag.."-data-"..i..".t7"
        print(ofile)
        torch.save(ofile,{inputs,labels})
    end

end
