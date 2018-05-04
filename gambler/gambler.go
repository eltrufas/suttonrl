package main

import (
	"fmt"
	"math"
)

const (
	theta    float64 = 10e-7
	headsP   float64 = 0.5
	discount float64 = 0.9
)

func intMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func intMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func getValue(s, a int, V *[101]float64) float64 {
	heads := intMin(s+a, 100)
	tails := intMax(s-a, 0)

	var rHeads float64
	if heads == 100 {
		rHeads = 1
	}

	val := headsP * (rHeads + discount*V[heads])
	val += (1 - headsP) * discount * V[tails]

	return val
}

func getActionRange(s int) (lower, upper int) {
	lower = 0
	upper = intMin(s, 100-s)
	return
}

func iterValueGreedy(s int, V *[101]float64) (float64, int) {
	lowerA, upperA := getActionRange(s)

	maxA := lowerA
	maxV := getValue(s, maxA, V)
	for a := lowerA + 1; a <= upperA; a++ {
		v := getValue(s, a, V)

		if v > maxV {
			maxA = a
			maxV = v
		}
	}

	return maxV, maxA
}

func main() {
	diff := 100.0
	var P [101]int
	var V [101]float64

	for diff > theta {
		diff = 0
		for s := 0; s <= 100; s++ {
			v := V[s]
			V[s], P[s] = iterValueGreedy(s, &V)

			diff = math.Max(diff, math.Abs(V[s]-v))
		}
	}

	for i := 0; i < 101; i++ {
		fmt.Printf("%v: %v\n", P[i], V[i])
	}

	fmt.Println("vim-go")
}
