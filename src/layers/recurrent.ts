
import { ILayer } from "./layer.interface"
import { tanh } from "../utils/activation"
import { clipValue } from "../utils/math"

export class SimpleRNN implements ILayer {
  private inputWeights: number[][] 
  private recurrentWeights: number[][] 
  private biases: number[] 
  private hiddenState: number[]
  private lastInputs: number[][] = []
  private lastHiddenStates: number[][] = []

  
  private adamT: number = 0
  private beta1: number = 0.9
  private beta2: number = 0.999
  private epsilon: number = 1e-8
  private mInputWeights: number[][]
  private vInputWeights: number[][]
  private mRecurrentWeights: number[][]
  private vRecurrentWeights: number[][]
  private mBiases: number[]
  private vBiases: number[]

  constructor(private inputSize: number, private hiddenSize: number) {
    this.inputWeights = this.initializeMatrix(hiddenSize, inputSize)
    this.recurrentWeights = this.initializeMatrix(hiddenSize, hiddenSize)
    this.biases = this.initializeVector(hiddenSize)
    this.hiddenState = Array(hiddenSize).fill(0)
    this.mInputWeights = this.initializeMatrix(hiddenSize, inputSize, () => 0)
    this.vInputWeights = this.initializeMatrix(hiddenSize, inputSize, () => 0)
    this.mRecurrentWeights = this.initializeMatrix(
      hiddenSize,
      hiddenSize,
      () => 0
    )
    this.vRecurrentWeights = this.initializeMatrix(
      hiddenSize,
      hiddenSize,
      () => 0
    )
    this.mBiases = Array(hiddenSize).fill(0)
    this.vBiases = Array(hiddenSize).fill(0)
  }

  forward(sequence: number[][]): number[][] {
    this.lastInputs = []
    this.lastHiddenStates = []
    this.hiddenState = Array(this.hiddenSize).fill(0);
    const outputs: number[][] = []
    for (let t = 0; t < sequence.length; t++) {
      const input = sequence[t]
      this.lastInputs.push(input)
      const inputPart = this.multiplyMatrixVector(this.inputWeights, input)
      const recurrentPart = this.multiplyMatrixVector(
        this.recurrentWeights,
        this.hiddenState
      )
      const summed = this.addVectors(
        this.addVectors(inputPart, recurrentPart),
        this.biases
      )
      this.hiddenState = summed.map(tanh)
      this.lastHiddenStates.push([...this.hiddenState])
      outputs.push([...this.hiddenState])
    }
    return outputs
  }

  backward(dOut: number[][], learningRate: number): number[][] {
    const clipThreshold = 1
    const T = this.lastInputs.length
    const dInputWeights = this.initializeMatrix(
      this.hiddenSize,
      this.inputSize,
      () => 0
    )
    const dRecurrentWeights = this.initializeMatrix(
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
      const dPre = dTotal.map((val, i) =>
        clipValue(val * dTanh[i], clipThreshold)
      )
      
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

    
    this.adamT++
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        const grad = dInputWeights[i][j] / T
        this.mInputWeights[i][j] =
          this.beta1 * this.mInputWeights[i][j] + (1 - this.beta1) * grad
        this.vInputWeights[i][j] =
          this.beta2 * this.vInputWeights[i][j] + (1 - this.beta2) * grad * grad
        const mHat =
          this.mInputWeights[i][j] / (1 - Math.pow(this.beta1, this.adamT))
        const vHat =
          this.vInputWeights[i][j] / (1 - Math.pow(this.beta2, this.adamT))
        this.inputWeights[i][j] -=
          (learningRate * mHat) / (Math.sqrt(vHat) + this.epsilon)
      }
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        const grad = dRecurrentWeights[i][j] / T
        this.mRecurrentWeights[i][j] =
          this.beta1 * this.mRecurrentWeights[i][j] + (1 - this.beta1) * grad
        this.vRecurrentWeights[i][j] =
          this.beta2 * this.vRecurrentWeights[i][j] +
          (1 - this.beta2) * grad * grad
        const mHat =
          this.mRecurrentWeights[i][j] / (1 - Math.pow(this.beta1, this.adamT))
        const vHat =
          this.vRecurrentWeights[i][j] / (1 - Math.pow(this.beta2, this.adamT))
        this.recurrentWeights[i][j] -=
          (learningRate * mHat) / (Math.sqrt(vHat) + this.epsilon)
      }
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      const grad = dBiases[i] / T
      this.mBiases[i] = this.beta1 * this.mBiases[i] + (1 - this.beta1) * grad
      this.vBiases[i] =
        this.beta2 * this.vBiases[i] + (1 - this.beta2) * grad * grad
      const mHat = this.mBiases[i] / (1 - Math.pow(this.beta1, this.adamT))
      const vHat = this.vBiases[i] / (1 - Math.pow(this.beta2, this.adamT))
      this.biases[i] -= (learningRate * mHat) / (Math.sqrt(vHat) + this.epsilon)
    }

    return dInputs
  }

  serialize(): any {
    return {
      inputWeights: this.inputWeights,
      recurrentWeights: this.recurrentWeights,
      biases: this.biases,
    }
  }

  load(serialized: any): void {
    this.inputWeights = serialized.inputWeights
    ;(this.recurrentWeights = serialized.recurrentWeights),
      (this.biases = serialized.biases)
  }

  private initializeMatrix(
    rows: number,
    cols: number,
    initFunc: () => number = () => Math.random() * 0.1
  ): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, initFunc)
    )
  }

  private initializeVector(
    size: number,
    initFunc: () => number = () => Math.random() * 0.1
  ): number[] {
    return Array.from({ length: size }, initFunc)
  }

  private multiplyMatrixVector(matrix: number[][], vector: number[]): number[] {
    return matrix.map((row) =>
      row.reduce((sum, val, idx) => sum + val * vector[idx], 0)
    )
  }

  private addVectors(vec1: number[], vec2: number[]): number[] {
    return vec1.map((val, idx) => val + vec2[idx])
  }
}
