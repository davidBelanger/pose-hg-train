local Batcher, parent = torch.class('Batcher')
    
function Batcher:__init(fileList,batchsize)
    self.dataFileIndex = 1
    self.dataIndex = 1
    self.mb = batchsize
    print("reading pre-cached data from: "..fileList)
    self.dataFiles = self:readList(fileList) 
    self.numProcessedFromFile = 0

end


function Batcher:getData()
    self.loadedData = self.loadedData or torch.load(self.dataFiles[self.dataFileIndex])

    if(self.dataIndex > self.loadedData[1]:size(1)) then 
        assert(self.numProcessedFromFile == self.loadedData[1]:size(1))
        self.loadedData = nil 
        self.dataFileIndex = dataFileIndex + 1
        if(dataFileIndex > #dataFiles) then dataFileIndex = 1 end
        local dataFile = dataFiles[dataFileIndex]
        self.loadedData = torch.load(dataFile)
        self.numProcessedFromFile = 0
        self.dataIndex = 1
    end

    local len = (self.dataIndex + self.mb < self.loadedData[1]:size(1)) and self.mb or (self.loadedData[1]:size(1) - self.dataIndex+1)
    local iptr = self.loadedData[1]:narrow(1,self.dataIndex,len)
    local lptr = self.loadedData[2]:narrow(1,self.dataIndex,len)
    self.cudaInput = self.cudaInput or iptr:cuda()
    self.cudaLabel = self.cudaLabel or lptr:cuda()
    self.cudaInput:resize(iptr:size()):copy(iptr)
    self.cudaLabel:resize(lptr:size()):copy(lptr)
    
    self.numProcessedFromFile = self.numProcessedFromFile + len

    self.dataIndex = self.dataIndex + self.mb
    return self.cudaInput, self.cudaLabel

end

function Batcher:readList(file)
    local tab = {}
    for l in io.lines(file) do
        table.insert(tab,l)
    end
    return tab
end