import { SimpleRNN } from "../layers/recurrent"
import { Dense, identity, dIdentity } from "../layers/dense"
import { BaseModel } from "./model.abstract"

export class RNNModel extends BaseModel {
  private rnnLayer: SimpleRNN
  private denseLayer: Dense
  public inputSize: number
  public hiddenSize: number
  public outputSize: number


  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    super()
    this.inputSize = inputSize
    this.hiddenSize = hiddenSize
    this.outputSize = outputSize
    this.rnnLayer = new SimpleRNN(inputSize, hiddenSize)
    this.denseLayer = new Dense(hiddenSize, outputSize, identity, dIdentity)
  }

  forward(sequence: number[][]): number[][] {
    const rnnOutputs = this.rnnLayer.forward(sequence)
    return this.denseLayer.forward(rnnOutputs)
  }

  train(
    sequence: number[][],
    target: number[][],
    learningRate: number
  ): number {
    const rnnOutputs = this.rnnLayer.forward(sequence)
    const denseOutput = this.denseLayer.forward(rnnOutputs)
    let loss = 0
    const dLoss = denseOutput.map((o, i) =>
      o.map((val, j) => {
        const diff = val - target[i][j]
        loss += diff * diff
        return diff
      })
    )
    loss /= denseOutput.length
    const gradDense = this.denseLayer.backward(dLoss, learningRate)
    this.rnnLayer.backward(gradDense, learningRate)
    return loss
  }

  serialize(): any {
    return {
      type: "RNNModel",
      architecture: {
        inputSize: this.inputSize,
        hiddenSize: this.hiddenSize,
        outputSize: this.outputSize
      },
      rnnLayer: this.rnnLayer.serialize(),
      denseLayer: this.denseLayer.serialize()
    }
  }
  load(serialized: any): void {
    if (serialized.architecture.inputSize !== this.inputSize ||
      serialized.architecture.hiddenSize !== this.hiddenSize ||
      serialized.architecture.outputSize !== this.outputSize) {
        throw new Error("Model architecture does not match the saved model!")
      }
      this.rnnLayer.load(serialized.rnnLayer)
      this.denseLayer.load(serialized.denseLayer)
  }
}
