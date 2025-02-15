import { ILayer } from "./layer.interface"
import {
  initializeMatrix,
  multiplyMatrixVector,
  addVectors,
  initializeVector,
  clipValue,
} from "../utils/math"

export const identity = (x: number) => x
export const dIdentity = (x: number) => 1

export function relu(x: number): number {
  return Math.max(0, x)
}
export function dRelu(x: number): number {
  return x > 0 ? 1 : 0
}

export class Dense implements ILayer {
  private weights: number[][] 
  private biases: number[]
  private lastInput: number[][] | null = null
  private lastPreActivations: number[][] | null = null

  private adamT: number = 0
  private beta1: number = 0.9
  private beta2: number = 0.999
  private epsilon: number = 1e-8
  private mWeights: number[][]
  private vWeights: number[][]
  private mBiases: number[]
  private vBiases: number[]

  constructor(
    private inputSize: number,
    private outputSize: number,
    private activation: (x: number) => number = relu,
    private dActivation: (x: number) => number = dRelu
  ) {
    this.weights = initializeMatrix(this.outputSize, this.inputSize)
    this.biases = initializeVector(this.outputSize)
    this.mWeights = initializeMatrix(this.outputSize, this.inputSize, () => 0)
    this.vWeights = initializeMatrix(this.outputSize, this.inputSize, () => 0)
    this.mBiases = Array(this.outputSize).fill(0)
    this.vBiases = Array(this.outputSize).fill(0)
  }

  forward(input: number[][]): number[][] {
    this.lastInput = input
    this.lastPreActivations = []
    return input.map((x) => {
      const weighted = multiplyMatrixVector(this.weights, x)
      const preActivation = addVectors(weighted, this.biases)
      this.lastPreActivations!.push(preActivation)
      return preActivation.map(this.activation)
    })
  }

  backward(dOut: number[][], learningRate: number): number[][] {
    if (!this.lastInput || !this.lastPreActivations)
      throw new Error("No forward pass input recorded.")

    const batchSize = this.lastInput.length
    const dWeights: number[][] = Array.from({ length: this.outputSize }, () =>
      Array(this.inputSize).fill(0)
    )
    const dBiases: number[] = Array(this.outputSize).fill(0)
    const dInput: number[][] = Array.from({ length: batchSize }, () =>
      Array(this.inputSize).fill(0)
    )

    for (let i = 0; i < batchSize; i++) {
      const inputVec = this.lastInput[i]
      const preActivation = this.lastPreActivations[i]
      const dPre = dOut[i].map((grad, j) => {
        const rawGrad = grad * this.dActivation(preActivation[j])
        return clipValue(rawGrad, 1) 
      })
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

    this.adamT++
    for (let j = 0; j < this.outputSize; j++) {
      for (let k = 0; k < this.inputSize; k++) {
        const grad = dWeights[j][k] / batchSize
        this.mWeights[j][k] =
          this.beta1 * this.mWeights[j][k] + (1 - this.beta1) * grad
        this.vWeights[j][k] =
          this.beta2 * this.vWeights[j][k] + (1 - this.beta2) * grad * grad
        const mHat =
          this.mWeights[j][k] / (1 - Math.pow(this.beta1, this.adamT))
        const vHat =
          this.vWeights[j][k] / (1 - Math.pow(this.beta2, this.adamT))
        this.weights[j][k] -=
          (learningRate * mHat) / (Math.sqrt(vHat) + this.epsilon)
      }
      const gradBias = dBiases[j] / batchSize
      this.mBiases[j] =
        this.beta1 * this.mBiases[j] + (1 - this.beta1) * gradBias
      this.vBiases[j] =
        this.beta2 * this.vBiases[j] + (1 - this.beta2) * gradBias * gradBias
      const mHatBias = this.mBiases[j] / (1 - Math.pow(this.beta1, this.adamT))
      const vHatBias = this.vBiases[j] / (1 - Math.pow(this.beta2, this.adamT))
      this.biases[j] -=
        (learningRate * mHatBias) / (Math.sqrt(vHatBias) + this.epsilon)
    }

    return dInput
  }

  serialize(): any {
    return {
      weights: this.weights,
      biases: this.biases
    }
  }

  load(serialized: any): void {
    this.weights = serialized.weights,
    this.biases = serialized.biases
  }
}
