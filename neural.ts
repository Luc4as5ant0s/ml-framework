interface NN {
  ws: matrix[]
  bs: matrix[]
  as: matrix[]
  layers: number
  architecture: number[]
}

type matrix = {
  rows: number
  cols: number
  data: Array<number>
}

const ins = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]

const outs = [[0], [1], [1], [0]]

function sigmoid(x: number): number {
  return 1 / (1 + Math.E ** -x)
}

function randArray(length: number, min: number, max: number): Array<number> {
  const arr: number[] = [];
  for (let i = 0; i < length; i++) {
    arr.push(Math.random() * (max - min) + min);
  }
  return arr;
}

function zeroArray(length: number): Array<number> {
  return Array.from({ length }, () => 0)
}

export function fillRand(nn: NN) {
  let l = nn.layers
  let w = nn.ws
  let b = nn.bs
  for (let i = 0; i < l - 1; i++) {
    nn.ws[i].data = randArray(w[i].rows * w[i].cols, -1, 1)
    nn.bs[i].data = randArray(b[i].rows * b[i].cols, -1, 1)
  }
}

function at(mat: matrix, row: number, col: number) {
  return row * mat.cols + col
}

function matAt(mat: matrix, row: number, col: number) {
  return mat.data[at(mat, row, col)]
}

function multiplyMatrix(a: matrix, b: matrix, dst: matrix): void {
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

function sumMatrix(a: matrix, b: matrix) {
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      a.data[at(a, i, j)] += matAt(b, i, j)
    }
  }
}

function activateMatrix(matrix: matrix) {
  for (let i = 0; i < matrix.rows; i++) {
    for (let j = 0; j < matrix.cols; j++) {
      matrix.data[at(matrix, i, j)] = sigmoid(matAt(matrix, i, j))
    }
  }
}

export function initNN(layers: number[]): NN {
  const nn: NN = {
    as: [],
    ws: [],
    bs: [],
    layers: 0,
    architecture: [],
  }
  nn.as[0] = {
    rows: 1,
    cols: layers[0],
    data: zeroArray(layers[0]),
  }
  for (let i = 1; i < layers.length; i++) {
    nn.ws[i - 1] = {
      rows: nn.as[i - 1].cols,
      cols: layers[i],
      data: zeroArray(nn.as[i - 1].cols * layers[i]),
    }
    nn.bs[i - 1] = {
      rows: 1,
      cols: layers[i],
      data: zeroArray(layers[i]),
    }
    nn.as[i] = {
      rows: 1,
      cols: layers[i],
      data: zeroArray(layers[i]),
    }
  }
  nn.layers = layers.length
  nn.architecture = layers
  return nn
}

function printMatrix(matrix: matrix): void {
  console.log("   [")
  for (let i = 0; i < matrix.rows; i++) {
    let str = "     "
    for (let j = 0; j < matrix.cols; j++) {
      str += `${matrix.data[at(matrix, i, j)]}, `
    }
    console.log(str)
  }
  console.log("   ]\n")
}

export function printNN(nn: NN): void {
  for (let i = 0; i < nn.layers - 1; i++) {
    console.log("Layer", i + 1)
    console.log(" ws: ")
    printMatrix(nn.ws[i])
    console.log(" bs: ")
    printMatrix(nn.bs[i])
  }
}

function getCost(nn: NN, ins: number[][], outs: number[][]): number {
  let cost = 0
  ins.forEach((input: number[], index: number) => {
    forward(nn, input)
    outs[index].forEach((output: number, outIndex) => {
      cost += (nn.as[nn.as.length - 1].data[outIndex] - output) ** 2
    })
  })
  return cost / ins.length
}

export function forward(nn: NN, input: number[]) {
  nn.as[0].data = input
  for (let i = 0; i < nn.layers - 1; i++) {
    multiplyMatrix(nn.as[i], nn.ws[i], nn.as[i + 1])
    sumMatrix(nn.as[i + 1], nn.bs[i])
    activateMatrix(nn.as[i + 1])
  }
}

export function finiteDiff(nn: NN, ins: number[][], outs: number[][], eps: number = 1e-3): NN {
  const g: NN = initNN(nn.architecture)
  const cost = getCost(nn, ins, outs)
  for (let i = 0; i < nn.layers - 1; i++) {
    for (let j = 0; j < nn.ws[i].rows; j++) {
      for (let k = 0; k < nn.ws[i].cols; k++) {
        const saved = matAt(nn.ws[i], j, k)
        nn.ws[i].data[at(nn.ws[i], j, k)] += eps
        g.ws[i].data[at(g.ws[i], j, k)] = (getCost(nn, ins, outs) - cost) / eps
        nn.ws[i].data[at(nn.ws[i], j, k)] = saved
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
  const g = initNN(nn.architecture)
  ins.forEach((input: number[], i: number) => {
    const out = outs[i]
    forward(nn, input)

    nn.architecture.forEach((layer: number, j) => {
      g.as[j].data = zeroArray(layer)
    })
    out.forEach((output: number, j: number) => {
      g.as[nn.layers - 1].data[j] = (nn.as[nn.layers - 1].data[j] - output)
    })

    for (let l = nn.layers - 1; l > 0; l--) {
      nn.as[l].data.forEach((a: number, j: number) => {
        const da = g.as[l].data[j]
        const dactf = a * (1 - a)
        g.bs[l - 1].data[j] += 2*da * dactf
        for (let k = 0; k < nn.as[l - 1].cols; k++) {
          const prevAct = nn.as[l - 1].data[k]
          const prevW = matAt(nn.ws[l - 1], k, j)
          g.ws[l - 1].data[at(g.ws[l-1], k, j)] += 2*da * dactf * prevAct
          g.as[l - 1].data[k] += 2*da * dactf * prevW
        }
      })
    }
  })

  for (let i = 0; i < g.layers - 1; i++) {
    for (let j = 0; j < g.ws[i].rows; j++) {
      for (let k = 0; k < g.ws[i].cols; k++) {
        g.ws[i].data[at(g.ws[i], j, k)] /= ins.length
      }
    }
    for (let k = 0; k < g.bs[i].cols; k++) {
      g.bs[i].data[k] /= ins.length
    }
  }

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


