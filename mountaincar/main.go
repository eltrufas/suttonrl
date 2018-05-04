package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	"./env"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"golang.org/x/image/colornames"
)

const (
	nTilings    = 40
	tilingRes   = 20
	plotXOrigin = 1100.0
	plotYOrigin = 360.0
	plotXScale  = 300.0
	plotYScale  = 4200.0
)

func renderMountainCar(mc *env.MountainCar, imd *imdraw.IMDraw) {
	leftBound := 0
	rightBound := 680
	step := 2

	prev := 250*math.Sin(3*-1.2) + 260

	imd.Color = colornames.Red

	for i := leftBound + step; i <= rightBound; i += step {
		y := 250*math.Sin(3*(float64(i)/400.0-1.2)) + 260

		imd.Push(pixel.V(float64(i-step), prev))
		imd.Push(pixel.V(float64(i), y))
		imd.Line(2)

		prev = y
	}

	x, dx := mc.GetRawState()

	realX := 400 * (x + 1.2)
	realY := 250*math.Sin(3*x) + 260

	angle := math.Atan(3 * math.Cos(3*x))

	transform := pixel.IM.Rotated(pixel.V(0, 0), angle).Moved(pixel.V(realX, realY))

	imd.Color = colornames.Brown

	imd.SetMatrix(transform)
	imd.Push(
		pixel.V(-40, 50),
		pixel.V(-40, 0),
		pixel.V(40, 0),
		pixel.V(40, 50),
	)
	imd.Polygon(0)

	imd.SetMatrix(pixel.IM)

	colors := []color.Color{
		colornames.Red,
		colornames.Green,
		colornames.Blue,
		colornames.Yellow,
		colornames.Cyan,
		colornames.Orange,
		colornames.Pink,
		colornames.Maroon,
		colornames.Honeydew,
		colornames.Azure,
		colornames.Aqua,
		colornames.Greenyellow,
		colornames.Grey,
		colornames.Beige,
		colornames.Bisque,
		colornames.Burlywood,
		colornames.Goldenrod,
		colornames.Lightpink,
		colornames.Wheat,
		colornames.Turquoise,
		colornames.Floralwhite,
	}

	colorIdx := 0
	tileW := 1.0 / tilingRes

	realTileW := tileW * plotXScale * 1.7
	realTileH := tileW * plotYScale * 0.14
	var offsetX, offsetY float64
	stepX := realTileW / nTilings
	stepY := realTileH / nTilings
	for i, val := range mc.GetState().([]bool) {
		if val {
			j := i % ((tilingRes + 1) * (tilingRes + 1))
			tx := plotXOrigin + float64(j%(tilingRes+1))*realTileW - offsetX - 1.2*plotXScale
			ty := plotYOrigin + float64(j/(tilingRes+1))*realTileH - offsetY - 0.07*plotYScale

			imd.Push(
				pixel.V(tx, ty),
				pixel.V(tx+realTileW, ty+realTileH),
			)

			imd.Color = colors[colorIdx]
			imd.Rectangle(1)

			offsetX += stepX
			offsetY += stepY
			colorIdx = (colorIdx + 1) % nTilings
		}
	}

	imd.Color = colornames.Black
	imd.Push(
		pixel.V(plotXOrigin+plotXScale*-1.2, plotYOrigin),
		pixel.V(plotXOrigin+plotXScale*0.5, plotYOrigin),
	)
	imd.Line(1)

	imd.Push(
		pixel.V(plotXOrigin, plotYOrigin+0.07*plotYScale),
		pixel.V(plotXOrigin, plotYOrigin-0.07*plotYScale),
	)
	imd.Line(1)

	imd.Color = colornames.Blue

	imd.Push(pixel.V(plotXOrigin+x*plotXScale, plotYOrigin+dx*plotYScale))

	imd.Circle(10, 0)
}

func AsyncTrain(initWs map[env.Action][]float64, maxIter int, c chan map[env.Action][]float64) {
	left := maxIter
	ws := env.CopyWeights(initWs)

	init := env.CreateMountainCarConstructor(env.CreateTileEncoder(nTilings, tilingRes))

	for left > 0 {
		nextBatch := 1
		if left < nextBatch {
			nextBatch = left
		}
		left -= nextBatch
		env.SarsaLambda(init, ws, nextBatch, 0.1/8, 0.6, 1)

		c <- env.CopyWeights(ws)
	}
}

const (
	costXStep float64 = 0.05
	costYStep float64 = 0.005
)

func renderCostToGo(vals []float64, imd *imdraw.IMDraw) {
	var i int
	baseColor := pixel.RGB(0.3, 1, 0.3)
	for y := -0.07; y < 0.07-costYStep; y += costYStep {
		for x := -1.2; x < 0.5-costXStep; x += costXStep {
			imd.Color = baseColor.Mul(pixel.Alpha(vals[i]))
			imd.Push(
				pixel.V(plotXOrigin+plotXScale*x, plotYOrigin+plotYScale*y),
				pixel.V(plotXOrigin+plotXScale*(x+costXStep), plotYOrigin+plotYScale*(y+costYStep)),
			)
			imd.Rectangle(0)
			i++
		}
	}
}

func getCostToGo(weights map[env.Action][]float64, mc *env.MountainCar, c chan []float64) {
	as := mc.LegalActions()
	enc := env.CreateTileEncoder(nTilings, tilingRes)

	costs := make([]float64, 0, 2000)

	var maxCost float64

	for y := -0.07; y < 0.07-costYStep; y += costYStep {
		for x := -1.2; x < 0.5-costXStep; x += costXStep {
			s := enc(x+costXStep/2, y+costYStep/2).([]bool)
			q := -env.MaxQ(weights, s, as)
			if q > maxCost {
				maxCost = q
			}
			costs = append(costs, q)
		}
	}

	if maxCost > 0 {
		for i, val := range costs {
			costs[i] = val / maxCost
		}
	}

	c <- costs
}

func run() {
	fmt.Println("Lesgo")
	cfg := pixelgl.WindowConfig{
		Title:  "Mountain Car",
		Bounds: pixel.R(0, 0, 1280, 720),
	}

	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}

	frametime, err := time.ParseDuration("16.66ms")
	if err != nil {
		panic(err)
	}

	ws := env.CreateMountainCarWeights(nTilings, tilingRes)
	trainChan := make(chan map[env.Action][]float64)
	costChan := make(chan []float64)

	go AsyncTrain(ws, 100000, trainChan)

	init := env.CreateMountainCarConstructor(env.CreateTileEncoder(nTilings, tilingRes))

	costs := make([]float64, 2000, 2000)

	mc := init().(*env.MountainCar)
	for !win.Closed() {
		target := time.Now().Add(frametime)
		imd := imdraw.New(nil)

		select {
		case newWs := <-trainChan:
			ws = newWs
			go getCostToGo(ws, mc, costChan)
		case newCost := <-costChan:
			costs = newCost
		default:
		}

		if mc.Terminal() {

			mc = init().(*env.MountainCar)
		}

		a := env.SelectActionEGreedy(ws, mc.GetState().([]bool), mc.LegalActions())

		mc.Transition(a)

		win.Clear(colornames.Aliceblue)

		renderCostToGo(costs, imd)
		renderMountainCar(mc, imd)

		imd.Draw(win)
		win.Update()

		dt := target.Sub(time.Now())
		time.Sleep(dt)
	}
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	pixelgl.Run(run)
}
