require 'torch'
require 'nn'
require 'nngraph'
require 'cudnn'
require 'module/normal3DConv'
require 'module/normal3DdeConv'

function build_3DBlock(prev_fDim,next_fDim,kernelSz)
    
    local module = nn.Sequential()
 
    module:add(cudnn.normal3DConv(prev_fDim,next_fDim,kernelSz,kernelSz,kernelSz,1,1,1,(kernelSz-1)/2,(kernelSz-1)/2,(kernelSz-1)/2,0,0.001))
    module:add(cudnn.VolumetricBatchNormalization(next_fDim))
    module:add(nn.ReLU(true))

    return module

end

function build_3DFEBlock(prev_fDim,next_fDim,kernelSz)
    
    local module = nn.Sequential()
 
    module:add(cudnn.normal3DConv(prev_fDim,next_fDim,kernelSz,kernelSz,kernelSz,1,1,1,(kernelSz-1)/2,(kernelSz-1)/2,(kernelSz-1)/2,0,0.001))
    module:add(cudnn.VolumetricBatchNormalization(next_fDim))

    return module

end

function build_3DResBlock(prev_fDim,next_fDim)
    
    local module = nn.Sequential()

    local concat = nn.ConcatTable()
    local resBranch = nn.Sequential()
    local skipCon = nn.Sequential()

    resBranch:add(cudnn.normal3DConv(prev_fDim,next_fDim,3,3,3,1,1,1,1,1,1,0,0.001))
    resBranch:add(cudnn.VolumetricBatchNormalization(next_fDim))
    resBranch:add(nn.ReLU(true))

    resBranch:add(cudnn.normal3DConv(next_fDim,next_fDim,3,3,3,1,1,1,1,1,1,0,0.001))
    resBranch:add(cudnn.VolumetricBatchNormalization(next_fDim))

    if prev_fDim == next_fDim then
        skipCon = nn.Identity()
    else
        skipCon:add(cudnn.normal3DConv(prev_fDim,next_fDim,1,1,1,1,1,1,0,0,0,0,0.001)):add(cudnn.VolumetricBatchNormalization(next_fDim))
    end
    
    concat:add(resBranch)
    concat:add(skipCon)

    module:add(concat)
    module:add(nn.CAddTable(true))
    module:add(nn.ReLU(true))

    return module

end

function build_3DpoolBlock(poolSz)
    
    local module = nn.VolumetricMaxPooling(poolSz,poolSz,poolSz,poolSz,poolSz,poolSz)
    return module

end

function build_3DupsampleBlock(prev_fDim,next_fDim,kernelSz,str)
    
    local module = nn.Sequential()
    
    module:add(cudnn.normal3DdeConv(prev_fDim,next_fDim,kernelSz,kernelSz,kernelSz,str,str,str,(kernelSz-1)/2,(kernelSz-1)/2,(kernelSz-1)/2,str-1,str-1,str-1,0,0.001))
    module:add(cudnn.VolumetricBatchNormalization(next_fDim))
    module:add(nn.ReLU(true))

    return module

end

function build_model()
	
    local module = nn.Sequential()

    concat1 = nn.ConcatTable()
    branch1 = nn.Sequential()

    branch1:add(build_3DpoolBlock(2))
    branch1:add(build_3DResBlock(32,64))

    concat2 = nn.ConcatTable()
    branch2 = nn.Sequential()

    branch2:add(build_3DpoolBlock(2))
    branch2:add(build_3DResBlock(64,128))
    branch2:add(build_3DResBlock(128,128))
    branch2:add(build_3DResBlock(128,128))
    branch2:add(build_3DupsampleBlock(128,64,2,2))

    concat2:add(branch2)
    concat2:add(build_3DResBlock(64,64))

    branch1:add(concat2)
    branch1:add(nn.CAddTable())
    
    branch1:add(build_3DResBlock(64,64))
    branch1:add(build_3DupsampleBlock(64,32,2,2))

    concat1:add(branch1)
    concat1:add(build_3DResBlock(32,32))

    module:add(concat1)
    module:add(nn.CAddTable())
    
    module:add(build_3DResBlock(32,32))
    module:add(build_3DBlock(32,32,1))
    module:add(build_3DBlock(32,32,1)) 

    return module

end
------------
function LSTM_3D(input_dim)
  local input = nn.Identity()()
  local prev_c = nn.Identity()()
  local prev_h = nn.Identity()()
  
  --preactivations
  i2h_model=build_3DFEBlock(input_dim, 4 * input_dim,7)
  h2h_model=build_3DFEBlock(input_dim, 4 * input_dim,7)
  i2h = i2h_model(input)
  h2h = h2h_model(prev_h)
  preactivations = nn.CAddTable()({i2h, h2h}) 
  
  -- in activation
  in_transform = nn.Tanh()(nn.Narrow(2,1,input_dim)(preactivations))

  -- all gates
  in_gate = nn.Sigmoid()(nn.Narrow(2,input_dim + 1,input_dim)(preactivations))

  out_gate = nn.Sigmoid()(nn.Narrow(2,2 * input_dim + 1,input_dim)(preactivations))

  forget_gate = nn.Sigmoid()(nn.Narrow(2,3 * input_dim + 1,input_dim)(preactivations))

  -- next cell_state
  c_input = nn.CMulTable()({in_gate, in_transform})
  c_forget = nn.CMulTable()({forget_gate, prev_c})

  next_c = nn.CAddTable()({c_input,c_forget})
  -- next hidden state (output)
  c_transform = nn.Tanh()(next_c)
  next_h = nn.CMulTable()({out_gate,c_transform})

  return nn.gModule({input,prev_c,prev_h},{next_c,next_h})
end
-----------
model = nn.Sequential()

model:add(build_3DBlock(inputDim,16,7))
model:add(build_3DpoolBlock(2))

model:add(build_3DResBlock(16,32))
model:add(build_3DResBlock(32,32))
model:add(build_3DResBlock(32,32))

model:add(build_model())
model:add(cudnn.normal3DConv(32,jointNum,1,1,1,1,1,1,0,0,0,0,0.001))

----------------------
lstm = LSTM_3D(inputDim)
cudnn.convert(lstm,cudnn)
lstm:cuda()
----------------------

cudnn.convert(model, cudnn)
model:cuda()
criterion = nn.MSECriterion()
criterion:cuda()
cudnn.fastest = true
cudnn.benchmark = true
----------------------
protos = {}
protos.model = model
protos.lstm = lstm
protos.criterion= criterion


