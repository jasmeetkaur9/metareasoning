import Node as nd
import numpy as np
import random
import os
# Import your game implementation here.
from env_file import MetaWorldEnv
from matplotlib import pyplot as plt
episodes = 1
rewards = []
moving_average = []
time_taken = 0.0
reward_e = 0
num_of_plans = 2
actions_per_plan = 2
max_planning_time = 6
deadline = 3
actions = [1, 2]
#env = MetaWorldEnv(num_of_plans, actions_per_plan, deadline, actions, max_planning_time)
#game = env
#print("done initializing the env")
#random.seed(40)

class MCTS:

    def __init__(self, Node, env, Verbose=False):
        self.root = Node
        self.main = Node
        self.verbose = Verbose
        self.game = env
        self.arr = set()

    def Selection(self):
        SelectedChild = self.root
        HasChild = False

        # Check if child nodes exist.
        if (len(SelectedChild.children) > 0):
            HasChild = True
        else:
            HasChild = False
        BestAction = 1
        while (HasChild):
            SelectedChild, BestAction = self.SelectChild(SelectedChild)
            if (len(SelectedChild.children) == 0):
                HasChild = False
        # SelectedChild.visits += 1.0

        if (self.verbose):
            print("\nSelected: ", self.game.get_state_from_id(SelectedChild.state))

        return SelectedChild,BestAction

    # -----------------------------------------------------------------------#
    # Description:
    #	Given a Node, selects the first unvisited child Node, or if all
    # 	children are visited, selects the Node with greatest UTC value.
    # Node	- Node from which to select child Node from.
    # -----------------------------------------------------------------------#
    def SelectChild(self, Node):
        if (len(Node.children) == 0):
            return Node,1


        index = 0
        random_child_unvisited = []
        choose_r = False
        for Child in Node.children:
            index = index + 1
            if (Child.visits > 0.0):
                continue
            else:
                random_child_unvisited.append((Child,index))
                choose_r = True
        if choose_r:
            Len = len(random_child_unvisited)
            i = np.random.randint(0, Len)
            Child, index = random_child_unvisited[i]
            if (self.verbose):
                print("Considered child", self.game.get_state_from_id(Child.state), "UTC: inf in", Len )
            return Child,index

        MaxWeight = 0.0
        best_action = 1
        index = 0
        for Child in Node.children:
            Weight = self.EvalUTC(Child)
            # Weight = Child.sputc
            if (self.verbose):
                print("Considered child:", self.game.get_state_from_id(Child.state), "UTC:", Weight, "Visits:",Child.visits)
            if (Weight > MaxWeight):
                MaxWeight = Weight
                SelectedChild = Child
                best_action = index+1
            index = index + 1
        return SelectedChild,best_action

    def Expansion(self, Leaf):
        if (self.IsTerminal((Leaf))):
            #print("Is Terminal.")
            return False
        elif (Leaf.visits == 0):
            return Leaf
        else:
            # Expand.
            if (len(Leaf.children) == 0):
                Children= self.EvalChildren(Leaf)
                for NewChild in Children:
                    if (np.all(NewChild.state == Leaf.state)):
                        continue
                    Leaf.AppendChild(NewChild)
            assert (len(Leaf.children) > 0), "Error"
            Child = self.SelectChildNode(Leaf)

        if (self.verbose):
            print("Expanded: ", self.game.get_state_from_id(Child.state))
        return Child

    def IsTerminal(self, Node):
        # Evaluate if node is terminal.
        if (self.game.done(Node.state)):
            return True
        else:
            return False

    # return False # Why is this here?

    # -----------------------------------------------------------------------#
    # Description:
    #	Evaluates all the possible children states given a Node state
    #	and returns the possible children Nodes.
    # Node	- Node from which to evaluate children.
    # -----------------------------------------------------------------------#
    def EvalChildren(self, Node):
        Children = []
        Actions = []
        for action in range(self.game.num_of_actions):
            next_state, reward = self.game.step2(Node.state, self.game.get_action_from_action_index(action))
            done = self.game.done(next_state)
            ChildNode = nd.Node(next_state)
            Children.append(ChildNode)
        return Children

    # -----------------------------------------------------------------------#
    # Description:
    #	Selects a child node randomly.
    # Node	- Node from which to select a random child.
    # -----------------------------------------------------------------------#
    def SelectChildNode(self, Node):
        # Randomly selects a child node.
        Len = len(Node.children)
        assert Len > 0, "Incorrect length"
        i = np.random.randint(0, Len)
        return Node.children[i]

    # -----------------------------------------------------------------------#
    # Description:
    #	Performs the simulation phase of the MCTS.
    # Node	- Node from which to perform simulation.
    # -----------------------------------------------------------------------#
    def Simulation(self, Node):
        CurrentState = Node.state
        # if(any(CurrentState) == False):
        #	return None
        if (self.verbose):
            print("Begin Simulation")

        Level = self.GetLevel(Node)
        Result = 0
        # Perform simulation.
        while True:
            action = random.choice(self.game.actions)
            CurrentState, reward = self.game.step2(CurrentState,action)
            Level += 1.0
            Result = Result + reward
            if (self.verbose):
                print("CurrentState:", self.game.get_state_from_id(CurrentState))
                #game.PrintTablesScores(CurrentState)
            if self.game.done(CurrentState) :
                oldState = CurrentState
                CurrentState, reward = self.game.step2(CurrentState, action)
                if reward > 0:
                    self.arr.add((self.game.get_state_from_id(oldState),reward))
                break
                Level += 1.0
                Result = Result + reward

        if (self.verbose):
            print("Value returned :", Result)

        return Result

    # -----------------------------------------------------------------------#
    # Description:
    #	Performs the backpropagation phase of the MCTS.
    # Node		- Node from which to perform Backpropagation.
    # Result	- Result of the simulation performed at Node.
    # -----------------------------------------------------------------------#
    def Backpropagation(self, Node, Result):
        # Update Node's weight.
        CurrentNode = Node
        CurrentNode.wins += Result
        CurrentNode.ressq += Result ** 2
        CurrentNode.visits += 1
        self.EvalUTC(CurrentNode)

        while (self.HasParent(CurrentNode)):
            # Update parent node's weight.
            CurrentNode = CurrentNode.parent
            CurrentNode.wins += Result
            CurrentNode.ressq += Result ** 2
            CurrentNode.visits += 1
            self.EvalUTC(CurrentNode)
            if self.verbose:
                print("Parent Cosidered : ", self.game.get_state_from_id(CurrentNode.state))

    # self.root.wins += Result
    # self.root.ressq += Result**2
    # self.root.visits += 1
    # self.EvalUTC(self.root)

    # -----------------------------------------------------------------------#
    # Description:
    #	Checks if Node has a parent..
    # Node - Node to check.
    # -----------------------------------------------------------------------#
    def HasParent(self, Node):
        if (Node.parent == None):
            return False
        else:
            return True

    # -----------------------------------------------------------------------#
    # Description:
    #	Evaluates the Single Player modified UTC. See:
    #	https://dke.maastrichtuniversity.nl/m.winands/documents/CGSameGame.pdf
    # Node - Node to evaluate.
    # -----------------------------------------------------------------------#
    def EvalUTC(self, Node):
        # c = np.sqrt(2)
        c = 1.41
        w = Node.wins
        n = Node.visits
        sumsq = Node.ressq
        if (Node.parent == None):
            t = Node.visits
        else:
            t = Node.parent.visits

        UTC = w / n + c * np.sqrt(np.log(t) / n)
        D = 10000.
        Modification = 0.0
        #Modification = np.sqrt((sumsq - n * (w / n) ** 2 + D) / n)
        # print "Original", UTC
        # print "Mod", Modification
        Node.sputc = UTC + Modification
        return Node.sputc

    def SelectBestAction(self, Node):
        if len(Node.children) == 0:
            return Node,1
        index = 0
        MaxWeight = 0.0
        best_action = 1
        index = 0
        for Child in Node.children:
            Weight = Child.visits
            # Weight = Child.sputc
            if (self.verbose):
                print("Considered child:", self.game.get_state_from_id(Child.state), "UTC:", Weight)
            if (Weight > MaxWeight):
                MaxWeight = Weight
                SelectedChild = Child
                best_action = index+1
            index = index + 1
        return SelectedChild,best_action
    # -----------------------------------------------------------------------#
    # Description:
    #	Gets the level of the node in the tree.
    # Node - Node to evaluate the level.
    # -----------------------------------------------------------------------#
    def GetLevel(self, Node):
        Level = 0.0
        while (Node.parent):
            Level += 1.0
            Node = Node.parent
        return Level

    # -----------------------------------------------------------------------#
    # Description:
    #	Prints the tree to file.
    # -----------------------------------------------------------------------#
    def PrintTree(self):
        f = open('Tree.txt', 'w')
        Node = self.root
        self.PrintNode(f, Node, "", False)
        f.close()

    # -----------------------------------------------------------------------#
    # Description:
    #	Prints the tree Node and its details to file.
    # Node			- Node to print.
    # Indent		- Indent character.
    # IsTerminal	- True: Node is terminal. False: Otherwise.
    # -----------------------------------------------------------------------#
    def PrintNode(self, file, Node, Indent, IsTerminal):
        file.write(Indent)
        if (IsTerminal):
            file.write("\-")
            Indent += "  "
        else:
            file.write("|-")
            Indent += "| "

        string = str(self.GetLevel(Node)) + ") (["
        # for i in Node.state.bins: # game specific (scrap)
        # 	string += str(i) + ", "
        string += str(self.game.get_state_from_id(Node.state))
        string += "], W: " + str(Node.wins) + ", N: " + str(Node.visits) + ", UTC: " + str(Node.sputc) + ") \n"
        file.write(string)

        for Child in Node.children:
            self.PrintNode(file, Child, Indent, self.IsTerminal(Child))

    def PrintResult(self, Result):
        filename = 'Results.txt'
        if os.path.exists(filename):
            append_write = 'a'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not

        f = open(filename, append_write)
        f.write(str(Result) + '\n')
        f.close()

    def getStatesReached(self):
        print(len(self.arr))

    # -----------------------------------------------------------------------#
    # Description:
    #	Runs the SP-MCTS.
    # MaxIter	- Maximum iterations to run the search algorithm.
    # -----------------------------------------------------------------------#
    def Run(self, MaxIter=5000, seed=40):
        action = 0
        if self.verbose:
            print("Search for best action")
        while True:
            X, action = self.Selection()
            if self.verbose:
                print("node : ", self.game.get_state_from_id(X.state))
            if self.IsTerminal(X):
                break
            Y = self.Expansion(X)
            if (Y):
                for k in range(1, MaxIter):
                    if self.verbose:
                        print("\n===== Begin iteration:", k, "=====")
                    Result = self.Simulation(Y)
                    self.Backpropagation(Y, Result)
            else:
                Result = self.game.reward_model[X.state]
                #print("Result: ", Result)
                self.Backpropagation(X, Result)
            #print(Result)
            #self.PrintResult(Result)
            break
        if self.verbose:
            print("Search complete.")
        _, BestAction = self.SelectBestAction(self.root)
        if self.verbose:
            print(BestAction, self.game.get_state_from_id(self.root.state))
            print("Root node : " , self.game.get_state_from_id(self.root.state))
        return BestAction


