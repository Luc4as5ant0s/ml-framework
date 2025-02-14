import { Dense } from "../layers/dense";
import { BaseModel } from "./model.abstract";

export class NeuralNetwork extends BaseModel {
  private layers: Dense[];

  constructor(layerSizes: number[]) {
    super();
    this.layers = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(new Dense(layerSizes[i], layerSizes[i + 1]));
    }
  }

  forward(input: number[][]): number[][] {
    let output = input;
    for (const layer of this.layers) {
      output = layer.forward(output);
    }
    return output;
  }

  train(input: number[][], target: number[][], learningRate: number): number {
    const output = this.forward(input);
    let loss = 0;
    const dLoss = output.map((o, i) =>
      o.map((val, j) => {
        const diff = val - target[i][j];
        loss += diff * diff;
        return diff; 
      })
    );
    loss /= output.length;
    let grad = dLoss;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad, learningRate);
    }
    return loss;
  }
}
