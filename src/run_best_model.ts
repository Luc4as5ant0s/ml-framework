import { RNNModel } from "./models/rnn"
import * as fs from "fs"
import * as path from "path"

const modelPath = path.join(__dirname, "best_model.json")

if (!fs.existsSync(modelPath)) {
  console.error("No saved best model found. Please run training first.")
  process.exit(1)
}

const savedData = JSON.parse(fs.readFileSync(modelPath, "utf-8"))
const bestLoss = savedData.bestLoss
console.log(`Loaded best model with loss: ${bestLoss}`)

const arch = savedData.model.architecture

const rnn = new RNNModel(arch.inputSize, arch.hiddenSize, arch.outputSize)

rnn.load(savedData.model)

const testSequence = [[0.5], [0.6], [0.7], [0.8], [0.9]]
const predictions = rnn.forward(testSequence)

console.log("Test sequence:", testSequence)
console.log("Predicted next values:", predictions)
