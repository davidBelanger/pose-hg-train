require 'paths'

package.path = package.path .. ';../../torch-util/?.lua'
package.path = package.path .. ';../../NLPConv/rnn/?.lua'

paths.dofile('../../spen/SPENPoseInterface.lua')

spenInterface = SPENPoseInterface()


paths.dofile('ref.lua')     -- Parse command line input and do global variable initialization
paths.dofile('data.lua')    -- Set up data processing
paths.dofile('model.lua')   -- Read in network model
paths.dofile('cachevaliddata.lua')   -- Load up training/testing functions

isFinished = false -- Finish early if validation accuracy plateaus, can be adjusted with opt.threshold

step('valid')
   