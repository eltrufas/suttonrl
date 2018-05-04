package main

import (
	"fmt"
	"math"
)

const (
	theta float64 = 1e-7 // Un numero muy pequeño
)

type JackPolicyIter struct {
	LocStates, MaxMornCars, MaxTrans int
	Discount                         float64         // Valor de descuento
	Policy                           [21][21]int     // La politica a aplicar
	V                                [21][21]float64 // Valor calculado de cada estado
	// Probablidades de pasar de un estado en la mañana a otro en la tarde. Uno por
	// cada locacion
	P1, P2 [26][21]float64
	// Recompensa esperada por dejar n carros en una ubicacion. Uno por cada ubicación.
	R1, R2 [26]float64
}

func factorial(n float64) float64 {
	if n > 0 {
		return (n * factorial(n-1.0))
	}

	return 1.0
}

func poisson(n int, lambda float64) float64 {
	return (math.Exp(-lambda) * math.Pow(lambda, float64(n)) / factorial(float64(n)))
}

func intMin(a, b int) int {
	if a > b {
		return b
	} else {
		return a
	}
}

func intMax(a, b int) int {
	if a < b {
		return b
	} else {
		return a
	}
}

func computeProbs(probs *[26][21]float64, rews *[26]float64, lRent, lRet float64) {
	for rent := 0; rent < 1000; rent++ {
		rentProb := poisson(rent, lRent)
		if rentProb <= theta {
			break
		}

		// Calcular las recompensas esperada para este numero de carros rentados
		for n := 0; n < 26; n++ {
			fullfilled := intMin(rent, n)
			rews[n] += 10 * rentProb * float64(fullfilled)

			// Castigo por estacionamiento
			/*if n > 10 {
				rews[n] -= 4
			}*/
		}

		for ret := 0; ret < 1000; ret++ {
			retProb := poisson(ret, lRet)
			if retProb <= theta {
				break
			}

			// Calcular probabilidades de pasar de un estado a otro
			for n := 0; n < 26; n++ {
				fullfilled := intMin(rent, n)

				newN := n + ret - fullfilled
				newN = intMin(intMax(newN, 0), 20)

				probs[n][newN] += rentProb * retProb
			}
		}
	}
}

func (j *JackPolicyIter) EvalPolicy() {
	diff := 10.0
	for diff > theta {
		diff = 0

		for n := 0; n < j.LocStates; n++ {
			for m := 0; m < j.LocStates; m++ {
				v := j.V[n][m]
				a := j.Policy[n][m]
				j.V[n][m] = j.getValue(n, m, a)
				diff = math.Max(diff, math.Abs(j.V[n][m]-v))
			}
		}
	}
}

func (j *JackPolicyIter) getValue(n, m, a int) float64 {
	a = intMax(intMin(a, n), -m)
	a = intMax(intMin(a, 5), -5)

	// Calcular el castigo por transportar carros
	val := -2 * math.Abs(float64(a))

	// Numero de carros que habra en la mañana
	mornN := n - a
	mornM := m + a

	// Un empleado carrea un carro gratis cada mañana
	/*if a < 0 {
		val += 2
	}*/

	// Agregamos los valores de los estados posibles
	for newN := 0; newN < j.LocStates; newN++ {
		for newM := 0; newM < j.LocStates; newM++ {
			val += j.P1[mornN][newN] * j.P2[mornM][newM] * (j.R1[mornN] + j.R2[mornM] + j.Discount*j.V[newN][newM])
		}
	}

	return val
}

func (j *JackPolicyIter) ImprovePolicy() bool {
	changed := false
	for n := 0; n < j.LocStates; n++ {
		for m := 0; m < j.LocStates; m++ {
			t := j.Policy[n][m]
			j.Policy[n][m] = j.findPolicyGreedy(n, m)
			if t != j.Policy[n][m] {
				changed = true
			}
		}
	}
	return changed
}

func (j *JackPolicyIter) findPolicyGreedy(n, m int) int {
	minA := intMax(-5, -m)
	maxA := intMin(5, n)

	a := minA
	bestA := minA
	bestV := j.getValue(n, m, a)
	for a := minA + 1; a <= maxA; a++ {
		v := j.getValue(n, m, a)
		if v > (bestV + 1e-10) {
			bestV = v
			bestA = a
		}
	}

	return bestA
}

func (j *JackPolicyIter) PrintPolicy() {
	fmt.Println("POLICY")
	for n := 0; n < j.LocStates; n++ {
		for m := 0; m < j.LocStates; m++ {
			fmt.Printf("%3v", j.Policy[n][m])
		}
		fmt.Println("")
	}
}

func main() {
	var j JackPolicyIter
	j.LocStates = 21
	j.MaxTrans = 5
	j.MaxMornCars = j.LocStates + j.MaxTrans
	j.Discount = 0.9

	// Parametros de las distribuciones de Poisson
	lamRentA := 3.0
	lamRentB := 4.0
	lamRetA := 3.0
	lamRetB := 2.0

	fmt.Println("Calculando probabilidades y recompensas")
	computeProbs(&j.P1, &j.R1, lamRentA, lamRetA)
	computeProbs(&j.P2, &j.R2, lamRentB, lamRetB)
	fmt.Println("Done")

	changed := true
	for changed {
		j.PrintPolicy()
		j.EvalPolicy()
		changed = j.ImprovePolicy()
	}

	j.PrintPolicy()
}
