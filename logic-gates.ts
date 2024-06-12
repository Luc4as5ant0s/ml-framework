import { initNN, fillRand, backprop, learn, forward, printNN } from "./neural"

const ins = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]
const outs = [[0], [1], [1], [0]]

function runModel() {
  const nn = initNN([2, 2, 1])
  fillRand(nn)
  for (let i = 0; i < 10000; i++) {
    const g = backprop(nn, ins, outs)
    learn(nn, g)
  }
  ins.forEach((input: number[]) => {
    forward(nn, input)
    console.log("result: ", nn.as[nn.layers - 1].data)
  })
  printNN(nn)
}

runModel()