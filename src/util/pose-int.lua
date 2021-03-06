-- Get prediction coordinates
predDim = {nParts,2}

criterion = nn.ParallelCriterion():add(nn.MSECriterion()):add(nn.MSECriterion())

-- Code to generate training samples from raw images.
function generateSample(set, idx)
    local pts = annot[set]['part']
    local c = annot[set]['center']
    local s = annot[set]['scale']
    local img = image.load(opt.dataDir .. '/images/' .. annot[set]['images'][idx])

    -- For single-person pose estimation with a centered/scaled figure
    local inp = crop(img, c[idx], s[idx], 0, opt.inputRes)
    local out = torch.zeros(nParts, opt.outputRes, opt.outputRes)
    for j = 1,nParts do
        if pts[idx][j][1] > 0 then -- Checks that there is a ground truth annotation
            drawGaussian(out[j], transform(pts[idx][j], c[idx], s[idx], 0, opt.outputRes), 1)
        end
    end

    return inp,out
end

function preprocess(input, label)
    return input, {label,label}
end

function postprocess(set, idx, output)
    local preds = getPreds(output[#output])
    return preds
end

function accuracy(output,label)
    local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    return heatmapAccuracy(output[#output],label[#output],nil,jntIdxs[opt.dataset])
end
