import metaNode as nd
import numpy as np
import random
import os

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


class MCTS:

    def __init__(self, Node, env, version1 ,verbose=False):
        self.root = Node
        self.main = Node
        self.verbose = verbose
        self.game = env
        self.arr = set()
        self.version1 = version1

    def Selection(self):
        SelectedChild = self.root
        HasChild = False

        # Check if child nodes exist.
        if len(SelectedChild.children) > 0:
            HasChild = True
        else:
            HasChild = False
        BestAction = 1
        while (HasChild):
            SelectedChild, BestAction = self.SelectChild(SelectedChild)
            if len(SelectedChild.children) == 0:
                HasChild = False

        if self.verbose:
            print("Selected: ", self.game.get_state_from_id(SelectedChild.state))

        return SelectedChild, BestAction

    def SelectChild(self, Node):
        if len(Node.children) == 0:
            return Node, 1

        index = 0
        random_child_unvisited = []
        choose_r = False
        for Child in Node.children:
            index = index + 1
            if Child.visits > 0.0:
                continue
            else:
                random_child_unvisited.append((Child, index))
                choose_r = True
        if choose_r:
            Len = len(random_child_unvisited)
            i = np.random.randint(0, Len)
            Child, index = random_child_unvisited[i]
            if self.verbose:
                print("Considered child", self.game.get_state_from_id(Child.state), "UTC: inf")
            return Child, index

        MaxWeight = 0.0
        best_action = 1
        index = 0
        for Child in Node.children:
            Weight = self.EvalUTC(Child)
            # Weight = Child.sputc
            if self.verbose:
                print("Considered child:", self.game.get_state_from_id(Child.state), "UTC:", Weight, "Visits:",
                      Child.visits)
            if Weight > MaxWeight:
                MaxWeight = Weight
                SelectedChild = Child
                best_action = index + 1
            index = index + 1
        return SelectedChild, best_action

    def Expansion(self, Leaf):
        if self.IsTerminal((Leaf)):
            return False
        elif Leaf.visits == 0:
            return Leaf
        else:
            # Expand.
            if len(Leaf.children) == 0:
                Children = self.EvalChildren(Leaf)
                for NewChild in Children:
                    if np.all(NewChild.state == Leaf.state):
                        continue
                    Leaf.AppendChild(NewChild)
            assert (len(Leaf.children) > 0), "Error"
            Child = self.SelectChildNode(Leaf)

        if self.verbose:
            print("Expanded: ", self.game.get_state_from_id(Child.state))
        return Child

    def IsTerminal(self, Node):
        # Evaluate if node is terminal.
        if self.game.done(Node.state):
            return True
        else:
            return False

    def EvalChildren(self, Node):
        Children = []
        Actions = []
        for action in range(self.game.num_of_actions):
            next_state, reward = self.game.step_next_state(Node.state, self.game.get_action_from_action_index(action))
            done = self.game.done(next_state)
            ChildNode = nd.Node(next_state)
            Children.append(ChildNode)
        return Children

    def SelectChildNode(self, Node):
        # Randomly selects a child node.
        Len = len(Node.children)
        assert Len > 0, "Incorrect length"
        i = np.random.randint(0, Len)
        return Node.children[i]

    def Simulation(self, Node):
        CurrentState = Node.state
        if self.verbose:
            print("Begin Simulation")

        Level = self.GetLevel(Node)
        Result = 0
        # Perform simulation.
        while True:
            action = random.choice(self.game.actions)
            CurrentState, reward = self.game.step_next_state(CurrentState, action)
            Level += 1.0
            if reward > 0:
                Result = Result + 1
            if self.verbose:
                print("CurrentState:", self.game.get_state_from_id(CurrentState))
            if self.game.done(CurrentState):
                oldState = CurrentState
                CurrentState, reward = self.game.step_next_state(CurrentState, action)
                if reward > 0:
                    self.arr.add((self.game.get_state_from_id(oldState)))
                Level += 1.0
                if reward > 0:
                    Result = Result + 1
                break

        if self.verbose:
            print("Value returned :", Result)
        return Result

    def Backpropagation(self, Node, Result):
        # Update Node's weight.
        if self.verbose:
            print("Back propagating")
        CurrentNode = Node
        CurrentNode.wins += Result
        CurrentNode.ressq += Result ** 2
        CurrentNode.visits += 1
        self.EvalUTC(CurrentNode)

        while self.HasParent(CurrentNode):
            if self.verbose:
                print("Has parent")
            CurrentNode = CurrentNode.parent
            CurrentNode.wins += Result
            CurrentNode.ressq += Result ** 2
            CurrentNode.visits += 1
            self.EvalUTC(CurrentNode)
            if self.verbose:
                print("Parent Considered : ", self.game.get_state_from_id(CurrentNode.state))

    def HasParent(self, Node):
        if Node.parent == None:
            return False
        else:
            return True

    def EvalUTC(self, Node):
        # c = np.sqrt(2)
        c = 4
        w = Node.wins
        n = Node.visits
        sumsq = Node.ressq
        if Node.parent is None:
            t = Node.visits
        else:
            t = Node.parent.visits

        UTC = w / n + c * np.sqrt(np.log(t) / n)
        Node.utc = UTC
        return Node.utc

    def SelectBestAction(self, Node):
        if len(Node.children) == 0:
            return Node, 1

        MaxWeight = 0.0
        best_action = 1
        index = 0
        for Child in Node.children:
            Weight = Child.visits
            if Weight > MaxWeight:
                MaxWeight = Weight
                SelectedChild = Child
                best_action = index + 1
            index = index + 1
        return SelectedChild, best_action

    def GetLevel(self, Node):
        Level = 0.0
        while Node.parent:
            Level += 1.0
            Node = Node.parent
        return Level

    def PrintNode(self, file, Node, Indent, IsTerminal):
        file.write(Indent)
        if IsTerminal:
            file.write("\-")
            Indent += "  "
        else:
            file.write("|-")
            Indent += "| "

        string = str(self.GetLevel(Node)) + ") (["
        # for i in Node.state.bins: # game specific (scrap)
        # 	string += str(i) + ", "
        string += str(self.game.get_state_from_id(Node.state))
        string += "], W: " + str(Node.wins) + ", N: " + str(Node.visits) + ", UTC: " + str(Node.utc) + ") \n"
        file.write(string)

        for Child in Node.children:
            self.PrintNode(file, Child, Indent, self.IsTerminal(Child))

    def getStatesReached(self):
        print(len(self.arr))

    def Run(self, MaxIter=5000, seed=40):
        action = 0
        self.arr.clear()
        for i in range(0, MaxIter):
            X, action = self.Selection()
            if self.verbose:
                print("Selected", self.game.get_state_from_id(X.state))
            Y = self.Expansion(X)
            if (Y):
                if self.verbose:
                    print("Expanded", self.game.get_state_from_id(Y.state))
                Result = self.Simulation(Y)
                self.Backpropagation(Y, Result)
            else:
                if self.version1 :
                    Result = self.game.reward_model[X.state]
                    self.Backpropagation(X, Result)
                else :
                    Result = self.game.reward_function(X.state)
                    self.Backpropagation(X, Result)
        # self.getStatesReached()
        if self.verbose:
            print("Search complete.")
        _, BestAction = self.SelectBestAction(self.root)
        if self.verbose:
            print("Best Action :", BestAction, "For node :", self.game.get_state_from_id(self.root.state))
            # print("Root node : " , self.game.get_state_from_id(self.root.state))
        return BestAction
