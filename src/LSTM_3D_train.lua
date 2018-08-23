require 'torch'
require 'nn'
require 'cudnn'
require 'cunn'
require 'optim'
require 'image'
require 'sys'
local model_utils=require 'model_utils'

-- get all above things into flattened parameters tensor

params, grad_params = model_utils.combine_all_parameters(protos.model, protos.lstm)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning '..name)
    if name == 'lstm' then
      clones[name] = model_utils.clone_many_times(proto, seq_length-1, not proto.parameters)
    else
      clones[name] = model_utils.clone_many_times(proto, seq_length, not proto.parameters)
    end
end

optimState = {
    learningRate = lr,
    learningRateDecay = 0.0,
    weightDecay = 0.0,
    momentum = 0.0,
    alpha = aph,
    epsilon = eps
}
optimMethod = optim.rmsprop
epoch = 0

function train()
 
    print('==> training start')
   
    tot_error = 0
    iter = 0
    

    
    for n, sample in DataLoad() do 
        
        inputs = sample[1] --voxelized depth map
        heatmaps = sample[2] --3D heatmap
        inputs = inputs:type('torch.CudaTensor')
        collectgarbage() 
     
        -- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
        local sz = inputs:size(2)
        local initstate_c = torch.zeros(sz,inputDim,croppedSz,croppedSz,croppedSz)
        initstate_c=initstate_c:type('torch.CudaTensor')
        local initstate_h = initstate_c:clone()

        -- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
        local dfinalstate_c = initstate_c:clone()
        local dfinalstate_h = initstate_c:clone()

        local feval = function(x)
            
            if x ~= params then
                params:copy(x)
            end
            grad_params:zero()
            ------------------- forward pass -------------------
            local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
            local lstm_h = {[0]=initstate_h} -- output values of LSTM
            local outputs = {}           -- softmax outputs
            local loss = 0
            for t=1,seq_length do
              -- we're feeding the *correct* things in here
              if t == 1 then
                outputs[t] = clones.model[t]:forward(inputs[t])
                lstm_c = {[1]=initstate_c}
                lstm_h = {[1]=inputs[t]}
              else
                lstm_c[t], lstm_h[t] = unpack(clones.lstm[t-1]:forward{inputs[t], lstm_c[t-1], lstm_h[t-1]})
                outputs[t] = clones.model[t]:forward(lstm_h[t])
              end
              err =clones.criterion[t]:forward(outputs[t],heatmaps[t])
              loss = loss + err
            end
            tot_error = tot_error + loss
            ------------------ backward pass -------------------
            local deinputs = {}                              -- d loss / d input embeddings
            local dlstm_c = {[seq_length]=dfinalstate_c}    -- internal cell states of LSTM
            local dlstm_h = {}                                  -- output values of LSTM
            for t =seq_length,1,-1 do
              -- backprop through err
              local doutput_t = clones.criterion[t]:backward(outputs[t],heatmaps[t])
              
              if t == seq_length then
                assert (dlstm_h[t] == nil)
                dlstm_h[t] = clones.model[t]:backward(lstm_h[t], doutput_t)
              else
                dlstm_h[t]:add(clones.model[t]:backward(lstm_h[t], doutput_t))
              end
              
              -- backprop through LSTM timestep
              if t ==1 then
                deinputs[t] = dlstm_h[t]
              else
               deinputs[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t-1]:backward({inputs[t], lstm_c[t-1], lstm_h[t-1]},{dlstm_c[t], dlstm_h[t]})) 
              end
              
            end

            iter = iter + 1
            -- clip gradient element-wise
            grad_params:clamp(-5, 5)
            --grad_params=grad_params:type('torch.CudaTensor')
            --loss =loss:cuda()
            --err = err:cuda()
            return loss,grad_params

        end
        params=params:type('torch.CudaTensor')
        optimMethod(feval, params, optimState)
        
        if iter % loss_display_interval == 0 then
            print("epoch: " .. epoch .. "/" .. epochLimit .. " batch: " ..  n*batchSz .. "/" .. trainSz .. " loss: " .. tot_error/iter)
            --print('trian_nci')
            --print(protos.model:get(1):get(1).bias)
            --print(protos.lstm:get(4):get(1).bias)
            tot_error = 0
            iter = 0
        end
    
    end

    epoch = epoch + 1    
  
end



