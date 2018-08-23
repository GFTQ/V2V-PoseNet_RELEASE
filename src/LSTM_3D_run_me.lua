require 'torch'
require 'cunn'
require 'torch'
require 'nn'
require 'cudnn'
require 'module/normal3DConv'
require 'module/normal3DdeConv'


torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(0)
cutorch.manualSeedAll(0)
math.randomseed(os.time())

dofile "config.lua"

if db == "ICVL" then
    dofile "./data/ICVL/data.lua"
elseif db == "NYU" then
    dofile "./data/NYU/data.lua"
elseif db == "MSRA" then
    dofile "./data/MSRA/data.lua"
elseif db == "side" or db == "top" then
    dofile "./data/ITOP/data.lua"
end

dofile "LSTM_util.lua"
dofile "LSTM_3D_model.lua"
dofile "LSTM_3D_train.lua"
dofile "LSTM_3D_test.lua"


print("db: " .. db .. " mode: " .. mode)
if mode == "train" then
    
    --resume training with saved model
    if resume == true then
        print("model loading...")
        model = torch.load(model_dir .. model_name)
        dofile "train.lua"
        epoch = resume_epoch
    end
    
    trainJointWorld, trainRefPt, trainName = load_data("train")
    testJointWorld, testRefPt, testName = load_data("test")
    
    --multi thread
    threads = init_thread(trainJointWorld,trainRefPt,trainName,db)
    --multi GPU
    if nGPU > 1 then
        protos = makeDataParallel(protos, gpu_table)
    end

    while epoch < epochLimit do
        train()
        test(testRefPt, testName)
        
        filename = paths.concat(model_dir, "epoch" .. tostring(epoch).."LSTM_3D", model_name)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        model:clearState()
        if nGPU == 1 then
            torch.save(filename, protos)
        else
            torch.save(filename, protos:get(1))
        end
        print('==> saved model to '..filename)
        collectgarbage()
    end
end


if mode == "test" then
    epoch =1
    print("model loading...")
    protos = torch.load(model_dir.. "epoch" .. tostring(epoch).."LSTM_3D/"..model_name)
    
    testJointWorld, testRefPt, testName = load_data("test")
    test(testRefPt, testName)

end

