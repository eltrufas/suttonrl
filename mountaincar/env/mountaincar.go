package env

import (
	"math"
	"math/rand"
)

type State interface{}
type Action interface{}

type MDPConstructor func() MDP

type MDP interface {
	GetState() State
	LegalActions() []Action
	Transition(Action) float64
	Terminal() bool
}

type StateEncoder func(float64, float64) State

func IdentityEncoder(x, dx float64) State {
	return []float64{x, dx}
}

func CreateTileEncoder(nTilings, tilingRes int) StateEncoder {
	tileW := 1.0 / float64(tilingRes)
	offset := tileW / float64(nTilings)
	tilesPerTiling := (tilingRes + 1) * (tilingRes + 1)
	var stride int = tilingRes + 1
	return func(x, dx float64) State {
		nx := (x + 1.2) / 1.7
		ny := (dx + 0.07) / 0.14
		s := make([]bool, tilesPerTiling*nTilings)
		for i := 0; i < nTilings; i++ {
			yIdx := int(ny * float64(tilingRes))
			xIdx := int(nx * float64(tilingRes))

			idx := tilesPerTiling*i + yIdx*stride + xIdx

			s[idx] = true

			nx += offset
			ny += offset
		}
		return s
	}
}

type MountainCar struct {
	x, dx   float64
	Encoder StateEncoder
}

func (mc *MountainCar) GetRawState() (float64, float64) {
	return mc.x, mc.dx
}

func (mc *MountainCar) GetState() State {
	return mc.Encoder(mc.x, mc.dx)
}

func (mc *MountainCar) Transition(a Action) float64 {
	direction := a.(float64)

	mc.dx += 0.001*direction - 0.0025*math.Cos(3*mc.x)

	if mc.dx < -0.07 {
		mc.dx = -0.07
	} else if mc.dx > 0.07 {
		mc.dx = 0.07
	}

	mc.x += mc.dx
	if mc.x <= -1.2 {
		mc.x = -1.2
		mc.dx = 0
	}

	if mc.x >= 0.5 {
		mc.x = 0.5
		return 0
	} else {
		return -1
	}
}

func (mc *MountainCar) Terminal() bool {
	return 0.5-mc.x < 1e-3
}

func (mc *MountainCar) LegalActions() []Action {
	as := []Action{0.0, 1.0, -1.0}

	j := rand.Intn(3)
	as[2], as[j] = as[j], as[2]
	j = rand.Intn(2)
	as[1], as[j] = as[j], as[1]

	return as
}

func CreateMountainCarConstructor(encoder StateEncoder) MDPConstructor {
	return func() MDP {
		var mc MountainCar
		mc.x = rand.Float64()*0.2 - 0.6
		mc.Encoder = encoder

		return &mc
	}
}

func Q(weights map[Action][]float64, state []bool, a Action) (q float64) {
	ws := weights[a]

	for i, val := range state {
		if val {
			q += ws[i]
		}
	}

	return
}

func MaxQ(weights map[Action][]float64, state []bool, as []Action) (maxQ float64) {
	maxQ = Q(weights, state, as[0])

	for _, a := range as[1:] {
		q := Q(weights, state, a)

		if q > maxQ {
			maxQ = q
		}
	}

	return
}

func SelectActionGreedy(weights map[Action][]float64, state []bool, as []Action) (maxA Action) {
	maxA = as[0]
	maxQ := Q(weights, state, as[0])

	for _, a := range as[1:] {
		q := Q(weights, state, a)

		if q > maxQ {
			maxA = a
			maxQ = q
		}
	}

	return
}

func SelectActionEGreedy(weights map[Action][]float64, state []bool, as []Action) Action {
	if rand.Float64() > 1 {
		idx := rand.Intn(len(as))
		return as[idx]
	} else {
		return SelectActionGreedy(weights, state, as)
	}
}

func QLearning(init MDPConstructor, weights map[Action][]float64, episodes int, alpha, discount float64) {
	for i := 0; i < episodes; i++ {
		mdp := init()
		s := mdp.GetState().([]bool)

		for !mdp.Terminal() {
			a := SelectActionEGreedy(weights, s, mdp.LegalActions())
			r := mdp.Transition(a)

			ws := weights[a]

			if mdp.Terminal() {
				change := alpha * (r - Q(weights, s, a))
				for i, val := range s {
					if val {
						ws[i] += change
					}
				}
				break
			}

			sPrime := mdp.GetState().([]bool)

			change := alpha * (r + discount*MaxQ(weights, sPrime, mdp.LegalActions()) - Q(weights, s, a))
			for i, val := range s {
				if val {
					ws[i] += change
				}
			}

			s = sPrime
		}
	}
}

func Sarsa(init MDPConstructor, weights map[Action][]float64, episodes int, alpha, discount float64) {
	var R float64
	for i := 0; i < episodes; i++ {
		mdp := init()
		s := mdp.GetState().([]bool)
		a := SelectActionEGreedy(weights, s, mdp.LegalActions())
		for !mdp.Terminal() {
			r := mdp.Transition(a)
			R += r
			ws := weights[a]

			if mdp.Terminal() {
				change := alpha * (r - Q(weights, s, a))
				for i, val := range s {
					if val {
						ws[i] += change
					}
				}
				break
			}

			sPrime := mdp.GetState().([]bool)
			aPrime := SelectActionEGreedy(weights, s, mdp.LegalActions())

			change := alpha * (r + discount*Q(weights, sPrime, aPrime) - Q(weights, s, a))
			for i, val := range s {
				if val {
					ws[i] += change
				}
			}

			a = aPrime
			s = sPrime
		}
	}
}

func SarsaLambda(init MDPConstructor, weights map[Action][]float64, episodes int, alpha, lambda, discount float64) {
	for i := 0; i < episodes; i++ {
		mdp := init()
		s := mdp.GetState().([]bool)
		a := SelectActionEGreedy(weights, s, mdp.LegalActions())
		zs := make([]float64, len(s))
		for !mdp.Terminal() {
			delta := mdp.Transition(a)
			ws := weights[a]
			for i, val := range s {
				if val {
					delta -= ws[i]
					zs[i]++
					if zs[i] < 1 {
						zs[i] = 1
					}
				}
			}
			if mdp.Terminal() {
				for i, z := range zs {
					ws[i] += alpha * delta * z
				}
				break
			}

			sPrime := mdp.GetState().([]bool)
			aPrime := SelectActionEGreedy(weights, s, mdp.LegalActions())

			wsPrime := weights[aPrime]
			for i, val := range sPrime {
				if val {
					delta += discount * wsPrime[i]
				}
			}
			for i, z := range zs {
				ws[i] += alpha * delta * z
				zs[i] *= discount * lambda
			}

			a = aPrime
			s = sPrime
		}
	}
}

func CreateMountainCarWeights(nTilings, tilingRes int) map[Action][]float64 {
	ws := make(map[Action][]float64)

	s := nTilings * (tilingRes + 1) * (tilingRes + 1)

	ws[1.0] = make([]float64, s)
	ws[0.0] = make([]float64, s)
	ws[-1.0] = make([]float64, s)

	return ws
}

func CopyWeights(ws map[Action][]float64) map[Action][]float64 {
	newWs := make(map[Action][]float64)

	s := len(ws[0.0])

	newWs[1.0] = make([]float64, s)
	newWs[0.0] = make([]float64, s)
	newWs[-1.0] = make([]float64, s)

	copy(newWs[-1.0], ws[-1.0])
	copy(newWs[0.0], ws[0.0])
	copy(newWs[1.0], ws[1.0])

	return newWs
}
