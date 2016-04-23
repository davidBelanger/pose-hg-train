--- Load up network model or initialize from scratch
paths.dofile('models/' .. opt.netType .. '.lua')

if(opt.createSPEN) then assert(opt.loadModel) end

-- Continuing an experiment where it left off
if opt.continue or opt.branch ~= 'none' then
    local prevModel = opt.load .. '/final_model.t7'
    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

    if(opt.createSPEN) then
        model = spenInterface:createSPENModel(model)
    end

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    model = createModel(modelArgs)
    assert(not opt.createRNN,'are you sure you want to be training the HG model from scratch')
end

-- Criterion (can be set in the opt.task file as well)
if not criterion then
    criterion = nn[opt.crit .. 'Criterion']()
end

if opt.GPU ~= -1 then
    -- Convert model to CUDA
    print('==> Converting model to CUDA')
    model:cuda()
    criterion:cuda()
end
