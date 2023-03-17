import numpy as np
from queue import PriorityQueue
from math import sqrt
import itertools
import time

PAIRWISEPRUNING = True
HEURISTICPRUNING = True
TIMEALLOWED = 120

class GraphVertex():
    """A graph vertex. The use of multiple can form a graph.
    
    #### Attributes
        name: str
            The name used to refer to a vertex, typically the carteasian coordiantes in the map
        neighbours: list
            A list of the neighbouring vertices
        
    #### Methods
        addNeighbour(neighbour):
            Adds a GraphVertex to the list of neighbours
        getNeighboursNames(): list
            Returns a list of all the names of the object's neighbours
        getNeighbours(): list
            Returns a list of all the objects neighbours
    """
    def __init__(self,name: str):
        self.name = name
        self.neighbours = []
        
    def __str__(self):
        return self.name

    def __lt__(self, other):
        return(int(self.name[0]) < int(other.name[0]))

    def addNeighbour(self,neighbour):
        #Adds a single neighbour to 
        if type(neighbour) == GraphVertex:
            #if neighbour not in self.neighbours:
            self.neighbours.append(neighbour)        
        elif type(neighbour) == list:
            for i in neighbour:
                self.addNeighbour(i)

        else:
            raise(TypeError("Unexpected type added as neighbour to GraphVertex",type(neighbour)))
    def getNeighboursNames(self):
        return [str(i) for i in self.neighbours]
    def getNeighbours(self):
        return self.neighbours


class Tree():
    """An abstract class that defines common attributes and methods for trees
    #### Attributes
    children: list
        A list of all trees branching off of this tree
    layer: int
        Defines how far down the tree the node is
    
    #### Methods
    addChild(child):
        Appends a child to the list of children
    getChildren(): list
        Returns the list of children
    removeChild(child):
        Searches the list of children and removes all instances of the child if it is found there
    """
    def __init__(self,children: list,layer):
        self.children = children
        self.layer = layer

    def addChild(self,child):
        self.children.append(child)

    def getChildren(self):
        return self.children
    
    def removeChild(self,child):
        for i in self.children:
            if i == child:
                self.children.remove(i)
class ICT(Tree):
    """The Increasing Cost Tree, used for the high-level algorithm in ICTS. Inherits from Tree
    ####Attributes
    contents: dict
        A dictionary that stores the costs for each agent
    
    ####Methods
    calculateRoot(startingVertices, targetVertices): dict
        Calculates the costs for each agent at the root of the tree
    
    
    """
    def __init__(self,layer):
        #super().__init__(self,children)
        self.children = []
        self.layer = layer

    def calculateRoot(self,startingVertices,targetVertices):
        minCosts = {}
        for i in startingVertices:
            minCosts[i] = self.aStar(startingVertices[i],targetVertices[i])
        self.contents = minCosts
        return minCosts
    def setContents(self,contents):
        self.contents = contents
    def aStar(self,start,goal):
        visited = []
        fringe = PriorityQueue()

        fringe.put((self.calculateHeuristic(start,goal),0,start))

        while not fringe.empty():
            (heuristic, cost, node) = fringe.get()
            if node not in visited:
                visited.append(node)

                if node == goal:
                    return(cost)
                for neighbour in node.getNeighbours():
                    if neighbour not in visited:
                        neighbourHeuristic = self.calculateHeuristic(neighbour,goal)
                        new_cost = cost + 1
                        fringe.put((neighbourHeuristic+new_cost,new_cost,neighbour))


    def calculateHeuristic(self,start,end):
        startingNode = str(start).split(" ")
        endNode = str(end).split(" ")
        #a^2+b^2=c^2
        return(sqrt(((int(startingNode[0])-int(endNode[0]))**2)+((int(startingNode[1])-int(endNode[1]))**2)))

    def getContents(self):
        return self.contents

class MDD(Tree):
    def __init__(self,contents,children,parents):
        #super().__init__(self,children)
        self.children = []
        if len(parents) > 0:
            self.layer = parents[0].layer +1
        else:
            self.layer = 0
        self.contents = contents
        self.parents = parents
    def addParent(self,newParent):
        self.parents.append(newParent)

    def getParents(self):
        return self.parents
    
    def getContents(self):
        return self.contents

class ICTS():
    def __init__(self,G,s,t):
        self.graph = G
        self.startingVertices = s
        self.targetVertices = t



    def highlevel(self):
        costTree = ICT(layer=0)
        costTree.calculateRoot(self.startingVertices,self.targetVertices)
        #print(costTree.getContents())
        if self.lowLevel(costTree.getContents()):
            return costTree.getContents()
        else:
            lastLayer = []
            lastLayer.append(costTree)
            layer = 1
            nextLayer = []
            while True:
                for tree in lastLayer:
                    costs = tree.getContents()
                    for i in self.startingVertices:
                        costs = {}
                        alreadyAdded = False
                        for j in self.startingVertices:
                            costs[j] = tree.getContents()[j]
                        costs[i] += 1
                        newTree = ICT(layer)
                        newTree.setContents(costs)
                        for k in nextLayer:
                            if k.getContents == costs:
                                alreadyAdded = True
                        if not alreadyAdded:
                            nextLayer.append(newTree)


                for tree in nextLayer:
                    if self.lowLevel(tree.getContents()):
                        return tree.getContents()
                    
                    if time.time() > startTime + TIMEALLOWED:
                        return None
                lastLayer = nextLayer.copy()
                layer += 1

    def lowLevel(self,costs):
        MDDs = {}
        for i in self.startingVertices:
            MDDs[i] = self.buildMDD(self.startingVertices[i],self.targetVertices[i],costs[i])
        
        timestep = 0


        if PAIRWISEPRUNING:
            pairs = list(itertools.combinations(MDDs,2))
            for pair in pairs:
                currentLayer = []
                pairCosts = {}
                pairMDDs = {}
                for i in pair:
                    pairCosts[i] = costs[i]
                    currentLayer.append(MDDs[i][timestep])
                    pairMDDs[i] = MDDs[i]
                maxCost = max(pairCosts.values())+1
                if not self.checkIfPossible(maxCost,pairMDDs,currentLayer,0):
                    return False
        #print(currentLayer)
        currentLayer = []
        for i in MDDs:
            currentLayer.append(MDDs[i][timestep])
        maxCost = max(costs.values())+1
        #print(maxCost)
        return self.checkIfPossible(maxCost,MDDs,currentLayer,0)


    def checkIfPossible(self,maxCost,MDDs,currentLayer,timestep):
        if timestep == maxCost:
            return True
        if time.time() > startTime + TIMEALLOWED:
            return False
        possibleCombinations = list(itertools.product(*currentLayer))
        #print(possibleCombinations)
        
        for combination in possibleCombinations:
            currentLayer = []
            if len(combination) == len(set(combination)):
                counter = 0
                for i in MDDs:
                    if timestep >= len(MDDs[i])-1:
                        currentLayer.append(MDDs[i][len(MDDs[i])-1])
                    else:
                        children = MDDs[i][timestep][combination[counter]].getChildren()
                        tempDict = {}
                        for child in children:
                            tempDict[str(child.getContents())] = child
                        currentLayer.append(tempDict)
                    counter += 1

                if self.checkIfPossible(maxCost,MDDs,currentLayer,timestep+1):
                    return True
                
            
            
        return False

        
        

    def buildMDD(self,start,target,cost):
        cost += 1
        #Forward pass, expanding all possible nodes that can be reached within the given cost
        newMDD = []
        newMDD.append({str(start):MDD(start,[],[])})
        for timestep in range(1,cost):
            newLayer = {}
            
            #Expand all the nodes in the previous layer
            for parent in newMDD[timestep-1]:
                if parent == str(target):
                    newLayer[parent] = newMDD[timestep-1][parent]
                else:
                    newChildren = newMDD[timestep-1][parent].getContents().getNeighbours()
                    newChildren.append(newMDD[timestep-1][parent].getContents())

                for child in newChildren:
                    if str(child) in newLayer:
                        newLayer[str(child)].addParent(newMDD[timestep-1][parent])
                    else:
                        if HEURISTICPRUNING:
                            if self.distanceBetween(child,target) <= (cost-timestep):
                                newLayer[str(child)] = MDD(child,[],[newMDD[timestep-1][parent]])
                        else:
                            newLayer[str(child)] = MDD(child,[],[newMDD[timestep-1][parent]])
                    try:
                        newMDD[timestep-1][parent].addChild(newLayer[str(child)])
                    except KeyError:
                        pass

            newMDD.append(newLayer)

        #Backwards pass to remove all nodes that don't end at the target
        output = []
        for timestep in range(cost):
            layer = newMDD[cost-timestep-1]
            layerCopy = {}
            for name,node in layer.items():
                layerCopy[name] = node

            for node in layerCopy:
                if layer[node].getChildren() == [] and node != str(target):
                    for parent in layer[node].getParents():
                        parent.removeChild(layer[node])
                    del layer[node]

            output.append(layer)
        output.reverse()
        return output
    
    def distanceBetween(self,child,target):
        childX, childY = str(child).split(" ")
        targetX, targetY = str(target).split(" ")
        distance = abs(int(childX) - int(targetX)) + abs(int(childY) - int(targetY))
        return distance


mapName = "maze-32-32-4.map"
mapFile = open(mapName,"r")
mapLines = mapFile.readlines()
mapFile.close()

height = int(mapLines[1].split(" ")[1])
width = int(mapLines[2].split(" ")[1])
mapLines = mapLines[4:]
#Create an empty 2d array of all the walkable spaces for an agent
mapArea = np.empty((height,width),dtype=GraphVertex)
#Run through the whole map and create a graph vertex for each
for row in range(height):
    for col in range(width):
        if mapLines[row][col] == ".":
            mapArea[row,col] = GraphVertex(str(str(row)+ " "+ str(col)))
            #Check for Neighbours
            #Check above
            if mapArea[row-1,col] != None:
                mapArea[row,col].addNeighbour(mapArea[row-1,col])
                mapArea[row-1,col].addNeighbour(mapArea[row,col])
            #Check Behind
            if mapArea[row,col-1] != None:
                mapArea[row,col].addNeighbour(mapArea[row,col-1])
                mapArea[row,col-1].addNeighbour(mapArea[row,col])

#Create a graph from the Vertices created
graph = {}
for i in mapArea:
    for j in i:
        if j != None:
            graph[str(j)] = j

scenarioName = "maze-32-32-4-even-1.scen"
scenFile = open(scenarioName,"r")
scenario = scenFile.readlines()
scenFile.close()

#Remove the first row that just states the version number
scenario = scenario[1:]
for NUMOFAGENTS in range(1,100):
    startingVertices = {}
    targetVertices = {}

    for i in range(NUMOFAGENTS):
        try:
            agent = scenario[i].split("\t")
            startingVertices[i] = graph[(agent[4] + " " + agent[5])]
            targetVertices[i] = graph[(agent[6]+ " " + agent[7])]
        except(KeyError):
            try:
                del startingVertices[i]
            except(KeyError):
                pass


    global startTime
    startTime = time.time()
    
    solver = ICTS(graph,startingVertices,targetVertices)
    output = solver.highlevel()
    print(NUMOFAGENTS)
    if output:
        print(output)
    else:
        print(output)
        break
