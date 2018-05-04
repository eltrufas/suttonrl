package mdp

type Action uint64
type State uint64

type MDP interface {
	IsLegal(Action) bool
	LegalActions() []Action
	State() State
	Transition(Action) float64
	Terminal() bool
}

type Agent interface {
	Policy(State) Action
	Q(State, Action) float64
	SetQ(State, Action, float64)
}

type MDPConstructor func() MDP

func BuildSarsaPolicy(c MDPConstructor, agent *Agent, alpha, discount float64, episodes int) {
	for i := 0; i < episodes; i++ {
		m := c()
		s := m.State()
		a := agent.Policy(s)
		for !m.Terminal() {
			r := m.Transition(a)
			sPrime := m.State()
			aPrime := agent.Policy(sPrime)
			newQ = a.Q(s, a) + alpha*(r+discount*agent.Q(sPrime, aPrime)-agent.Q(s, a))
			agent.setQ(s, a, newQ)
			s = sPrime
			a = aPrime
		}
	}
}
