import { SimpleRNN } from "../layers/recurrent";
import { Dense, identity, dIdentity } from "../layers/dense";
import { BaseModel } from "./model.abstract";

export class RNNModel extends BaseModel {
  private rnnLayer: SimpleRNN;
  private denseLayer: Dense;

  constructor(inputSize: number, hiddenSize: number, outputSize: number) {
    super();
    this.rnnLayer = new SimpleRNN(inputSize, hiddenSize);
    this.denseLayer = new Dense(hiddenSize, outputSize, identity, dIdentity);
  }

  forward(sequence: number[][]): number[][] {
    const rnnOutputs = this.rnnLayer.forward(sequence);
    return this.denseLayer.forward(rnnOutputs);
  }

  train(sequence: number[][], target: number[][], learningRate: number): number {
    const rnnOutputs = this.rnnLayer.forward(sequence);
    const denseOutput = this.denseLayer.forward(rnnOutputs);
    let loss = 0;
    const dLoss = denseOutput.map((o, i) =>
      o.map((val, j) => {
        const diff = val - target[i][j];
        loss += diff * diff;
        return diff;
      })
    );
    loss /= denseOutput.length;
    const gradDense = this.denseLayer.backward(dLoss, learningRate);
    this.rnnLayer.backward(gradDense, learningRate);
    return loss;
  }
}
