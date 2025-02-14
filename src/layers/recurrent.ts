import { ILayer } from "./layer.interface"
import { tanh } from "../utils/activation"
import {
  initializeMatrix,
  initializeVector,
  multiplyMatrixVector,
  addVectors,
} from "../utils/math"
export class SimpleRNN implements ILayer {
  private inputWeights: number[][] 
  private recurrentWeights: number[][] 
  private biases: number[]
  private hiddenState: number[]
  private lastInputs: number[][] = []
  private lastHiddenStates: number[][] = []

  constructor(private inputSize: number, private hiddenSize: number) {
    this.inputWeights = initializeMatrix(hiddenSize, inputSize)
    this.recurrentWeights = initializeMatrix(hiddenSize, hiddenSize)
    this.biases = initializeVector(hiddenSize)
    this.hiddenState = Array(hiddenSize).fill(0)
  }

  forward(sequence: number[][]): number[][] {
    this.lastInputs = []
    this.lastHiddenStates = []
    const outputs: number[][] = []
    for (let t = 0; t < sequence.length; t++) {
      const input = sequence[t]
      this.lastInputs.push(input)
      const inputPart = multiplyMatrixVector(this.inputWeights, input)
      const recurrentPart = multiplyMatrixVector(
        this.recurrentWeights,
        this.hiddenState
      )
      const summed = addVectors(
        addVectors(inputPart, recurrentPart),
        this.biases
      )
      this.hiddenState = summed.map(tanh)
      this.lastHiddenStates.push([...this.hiddenState])
      outputs.push([...this.hiddenState])
    }
    return outputs
  }

  backward(dOut: number[][], learningRate: number): number[][] {
    const T = this.lastInputs.length
    const dInputWeights = initializeMatrix(
      this.hiddenSize,
      this.inputSize,
      () => 0
    )
    const dRecurrentWeights = initializeMatrix(
      this.hiddenSize,
      this.hiddenSize,
      () => 0
    )
    const dBiases = Array(this.hiddenSize).fill(0)
    const dInputs: number[][] = Array.from({ length: T }, () =>
      Array(this.inputSize).fill(0)
    )
    let dHiddenNext = Array(this.hiddenSize).fill(0)

    for (let t = T - 1; t >= 0; t--) {
      const dTotal = dOut[t].map((val, i) => val + dHiddenNext[i])
      const dTanh = this.lastHiddenStates[t].map((h) => 1 - h * h)
      const dPre = dTotal.map((val, i) => val * dTanh[i])
      for (let i = 0; i < this.hiddenSize; i++) {
        dBiases[i] += dPre[i]
      }
      const input = this.lastInputs[t]
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.inputSize; j++) {
          dInputWeights[i][j] += dPre[i] * input[j]
        }
      }
      const prevHidden =
        t > 0 ? this.lastHiddenStates[t - 1] : Array(this.hiddenSize).fill(0)
      for (let i = 0; i < this.hiddenSize; i++) {
        for (let j = 0; j < this.hiddenSize; j++) {
          dRecurrentWeights[i][j] += dPre[i] * prevHidden[j]
        }
      }
      for (let j = 0; j < this.inputSize; j++) {
        let sum = 0
        for (let i = 0; i < this.hiddenSize; i++) {
          sum += this.inputWeights[i][j] * dPre[i]
        }
        dInputs[t][j] = sum
      }
      dHiddenNext = Array(this.hiddenSize).fill(0)
      for (let j = 0; j < this.hiddenSize; j++) {
        let sum = 0
        for (let i = 0; i < this.hiddenSize; i++) {
          sum += this.recurrentWeights[i][j] * dPre[i]
        }
        dHiddenNext[j] = sum
      }
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.inputWeights[i][j] -= learningRate * (dInputWeights[i][j] / T)
      }
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.recurrentWeights[i][j] -=
          learningRate * (dRecurrentWeights[i][j] / T)
      }
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      this.biases[i] -= learningRate * (dBiases[i] / T)
    }
    return dInputs
  }
}
