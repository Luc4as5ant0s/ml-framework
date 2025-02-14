export abstract class BaseModel {
  abstract forward(input: any): any;
  abstract train(input: any, target: any, learningRate: number): number;
}
