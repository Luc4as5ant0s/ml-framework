type Matrix = {
  rows: number
  cols: number
  data: number[]
}

type NN = {
  layers: number
  architecture: number[]
  as: Matrix[]
  ws: Matrix[]
  bs: Matrix[]
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.E ** -x)
}

function createMatrix(rows: number, cols: number, value: number = 0): Matrix {
  return {
    rows,
    cols,
    data: Array(rows * cols).fill(value),
  }
}

function randomMatrix(rows: number, cols: number): Matrix {
  return {
    rows,
    cols,
    data: Array.from({ length: rows * cols }, () => Math.random() * 2 - 1),
  }
}

function multiplyMatrix(a: Matrix, b: Matrix, dst: Matrix): void {
  if (a.cols !== b.rows) throw Error("Columns should equal rows")
  if (dst.rows !== a.rows) throw Error("Dst rows differ")
  if (dst.cols !== b.cols) throw Error("Dst cols differ")

  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      dst.data[at(dst, i, j)] = 0
      for (let k = 0; k < a.cols; k++) {
        dst.data[at(dst, i, j)] += matAt(a, i, k) * matAt(b, k, j)
      }
    }
  }
}

function sumMatrix(a: Matrix, b: Matrix) {
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      a.data[at(a, i, j)] += matAt(b, i, j)
    }
  }
}

function activateMatrix(matrix: Matrix) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.cols; j++) {
      matrix.data[at(matrix, i, j)] = sigmoid(matAt(matrix, i, j))
    }
  }
}

function matAt(m: Matrix, row: number, col: number): number {
  return m.data[row * m.cols + col]
}

function at(mat: Matrix, row: number, col: number) {
  return row * mat.cols + col
}

export function forward(nn: NN, input: number[]): number[] {
  nn.as[0].data = input
  for (let i = 0; i < nn.layers - 1; i++) {
    multiplyMatrix(nn.as[i], nn.ws[i], nn.as[i + 1])
    sumMatrix(nn.as[i + 1], nn.bs[i])
    activateMatrix(nn.as[i + 1])
  }
  return nn.as[nn.layers - 1].data
}

function zeroArray(length: number): number[] {
  return Array(length).fill(0)
}

export function initNN(architecture: number[]): NN {
  const layers = architecture.length
  const as = architecture.map((layer) => createMatrix(1, layer))
  const ws = architecture
    .slice(0, -1)
    .map((layer, i) => randomMatrix(layer, architecture[i + 1]))
  const bs = architecture.slice(1).map((layer) => createMatrix(1, layer))
  return { layers, architecture, as, ws, bs }
}

function zeroNN(architecture: number[]): NN {
  const layers = architecture.length
  const as = architecture.map((layer) => createMatrix(1, layer))
  const ws = architecture
    .slice(0, -1)
    .map((layer, i) => createMatrix(layer, architecture[i + 1]))
  const bs = architecture.slice(1).map((layer) => createMatrix(1, layer))
  return { layers, architecture, as, ws, bs }
}

function getCost(nn: NN, ins: number[][], outs: number[][]): number {
  return (
    ins.reduce((sum, input, i) => {
      const output = forward(nn, input)
      const target = outs[i]
      return (
        sum +
        output.reduce((cost, out, j) => {
          return cost + (out - target[j]) ** 2
        }, 0)
      )
    }, 0) / ins.length
  )
}

export function numGrad(nn: NN, ins: number[][], outs: number[][]): NN {
  const eps = 1e-4
  const g = initNN(nn.architecture)
  const cost = getCost(nn, ins, outs)

  for (let i = 0; i < nn.layers - 1; i++) {
    for (let j = 0; j < nn.ws[i].rows; j++) {
      for (let k = 0; k < nn.ws[i].cols; k++) {
        const saved = matAt(nn.ws[i], j, k)
        nn.ws[i].data[j * nn.ws[i].cols + k] += eps
        g.ws[i].data[j * g.ws[i].cols + k] =
          (getCost(nn, ins, outs) - cost) / eps
        nn.ws[i].data[j * nn.ws[i].cols + k] = saved
      }
    }

    for (let k = 0; k < nn.bs[i].cols; k++) {
      const saved = nn.bs[i].data[k]
      nn.bs[i].data[k] += eps
      g.bs[i].data[k] = (getCost(nn, ins, outs) - cost) / eps
      nn.bs[i].data[k] = saved
    }
  }

  return g
}

export function backprop(nn: NN, ins: number[][], outs: number[][]): NN {
  const g = zeroNN(nn.architecture)
  let cost = 0
  ins.forEach((input, i) => {
    const out = outs[i]
    forward(nn, input)

    nn.architecture.forEach((layer, j) => {
      g.as[j].data = zeroArray(layer)
    })

    out.forEach((output, j) => {
      const diff = nn.as[nn.layers - 1].data[j] - output
      g.as[nn.layers - 1].data[j] = diff
      cost += diff ** 2
      g.as[nn.layers - 1].data[j] = nn.as[nn.layers - 1].data[j] - output
    })

    for (let l = nn.layers - 1; l > 0; l--) {
      nn.as[l].data.forEach((a, j) => {
        const da = g.as[l].data[j]
        const dactf = a * (1 - a)
        g.bs[l - 1].data[j] += 2 * da * dactf

        for (let k = 0; k < nn.as[l - 1].cols; k++) {
          const prevAct = nn.as[l - 1].data[k]
          const prevW = matAt(nn.ws[l - 1], k, j)
          g.ws[l - 1].data[k * g.ws[l - 1].cols + j] += 2 * da * dactf * prevAct
          g.as[l - 1].data[k] += 2 * da * dactf * prevW
        }
      })
    }
  })

  for (let i = 0; i < g.layers - 1; i++) {
    for (let j = 0; j < g.ws[i].rows; j++) {
      for (let k = 0; k < g.ws[i].cols; k++) {
        g.ws[i].data[j * g.ws[i].cols + k] /= ins.length
      }
    }

    for (let k = 0; k < g.bs[i].cols; k++) {
      g.bs[i].data[k] /= ins.length
    }
  }
  console.log(cost / ins.length)
  return g
}

export function learn(nn: NN, g: NN, rate: number = 1) {
  for (let i = 0; i < nn.layers - 1; i++) {
    for (let j = 0; j < nn.ws[i].rows; j++) {
      for (let k = 0; k < nn.ws[i].cols; k++) {
        nn.ws[i].data[j * nn.ws[i].cols + k] -=
          rate * g.ws[i].data[j * g.ws[i].cols + k]
      }
    }

    for (let k = 0; k < nn.bs[i].cols; k++) {
      nn.bs[i].data[k] -= rate * g.bs[i].data[k]
    }
  }
}

export function printNN(nn: NN) {
  console.log("Neural Network Structure:")
  console.log("Layers:", nn.layers)
  console.log("Architecture:", nn.architecture)

  console.log("\nActivations (as):")
  nn.as.forEach((matrix, index) => {
    console.log(` Layer ${index}:`, matrix)
  })

  console.log("\nWeights (ws):")
  nn.ws.forEach((matrix, index) => {
    console.log(` Layer ${index + 1} Weights:`)
    console.log(`  [`)
    for (let i = 0; i < matrix.rows; i++) {
      let row: string[] = []
      for (let j = 0; j < matrix.cols; j++) {
        row.push(matAt(matrix, i, j).toFixed(4) + ",")
      }
      console.log("   ", row.join(" "))
    }
    console.log(`  ]`)
  })

  console.log("\nBiases (bs):")
  nn.bs.forEach((matrix, index) => {
    let biases: string[] = []
    for (let i = 0; i < matrix.cols; i++) {
      biases.push(matrix.data[i].toFixed(4))
    }
    console.log(` Layer ${index + 1} Biases: [${biases.join(", ")}]`)
  })
}
