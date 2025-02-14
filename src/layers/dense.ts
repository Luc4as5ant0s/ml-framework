import { ILayer } from "./layer.interface"
import {
  initializeMatrix,
  multiplyMatrixVector,
  addVectors,
  initializeVector,
} from "../utils/math"
import { relu } from "../utils/activation"

export class Dense implements ILayer {
  private weights: number[][]
  private biases: number[]
  private lastInput: number[][] | null = null

  constructor(private inputSize: number, private outputSize: number) {
    this.weights = initializeMatrix(this.outputSize, this.inputSize)
    this.biases = initializeVector(this.outputSize)
  }

  forward(input: number[][]): number[][] {
    this.lastInput = input
    return input.map((x) => {
      const weighted = multiplyMatrixVector(this.weights, x)
      const summed = addVectors(weighted, this.biases)
      return summed.map(relu)
    })
  }

  backward(dOut: number[][], learningRate: number): number[][] {
    if (!this.lastInput) throw new Error("No forward pass input recorded.")
    const batchSize = this.lastInput.length
    const dWeights: number[][] = Array.from({ length: this.outputSize }, () =>
      Array(this.inputSize).fill(0)
    )
    const dBiases: number[] = Array(this.outputSize).fill(0)
    const dInput: number[][] = Array.from({ length: batchSize }, () =>
      Array(this.inputSize).fill(0)
    )

    const dReLU = (x: number) => (x > 0 ? 1 : 0)

    for (let i = 0; i < batchSize; i++) {
      const inputVec = this.lastInput[i]
      const weighted = multiplyMatrixVector(this.weights, inputVec)
      const preActivation = addVectors(weighted, this.biases)
      const dPre = dOut[i].map((val, j) => val * dReLU(preActivation[j]))
      for (let j = 0; j < this.outputSize; j++) {
        for (let k = 0; k < this.inputSize; k++) {
          dWeights[j][k] += dPre[j] * inputVec[k]
        }
        dBiases[j] += dPre[j]
      }
      for (let k = 0; k < this.inputSize; k++) {
        let sum = 0
        for (let j = 0; j < this.outputSize; j++) {
          sum += this.weights[j][k] * dPre[j]
        }
        dInput[i][k] = sum
      }
    }

    for (let j = 0; j < this.outputSize; j++) {
      for (let k = 0; k < this.inputSize; k++) {
        this.weights[j][k] -= learningRate * (dWeights[j][k] / batchSize)
      }
      this.biases[j] -= learningRate * (dBiases[j] / batchSize)
    }
    return dInput
  }
}
