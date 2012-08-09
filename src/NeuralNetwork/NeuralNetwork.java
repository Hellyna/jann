package NeuralNetwork;

import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.security.SecureRandom;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

class NeuralNetwork {

    private static Calendar calendar = Calendar.getInstance();
    private static SimpleDateFormat dateFormat = 
            new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss");
    private static SecureRandom prng = new SecureRandom();
    private long rounds;
    private double learningRate;
    private int[] config;
    private int[] weightRange;
    private ArrayList<double[]> targetInputs;
    private ArrayList<double[]> targetOutputs;
    private ArrayList<double[][]> weights;
    private ArrayList<double[][]> previousWeights;
    private ArrayList<double[]> outputs;
    private ArrayList<double[]> deltas;
    
    NeuralNetwork(String path) {
        
    }

    NeuralNetwork(
            int[] config,
            ArrayList<double[]> targetInput,
            ArrayList<double[]> targetOutput,
            int[] weightRange,
            double learningRate,
            long rounds)
            throws ImpossibleNeuralConfigException,
            InvalidTargetOutputException,
            InvalidTargetInputException, 
            ImpossibleWeightRangeException {
        // Setting parameters in object for later access.
        this.config = config;
        this.targetInputs = targetInput;
        this.targetOutputs = targetOutput;
        this.learningRate = learningRate;
        this.weightRange = weightRange;
        this.verify();
        this.rounds = rounds;
        // Setting the neuron datas.
        this.weights = new ArrayList<>();
        this.previousWeights = new ArrayList<>();
        this.outputs = new ArrayList<>();
        this.deltas = new ArrayList<>();
        for (int i = 0; i < config.length; i++) {
            if (i == 0) {
                double[] temp2 = new double[config[i]];
                for (int j = 0; j < temp2.length; j++) {
                    temp2[j] = 0.0;
                }
                this.outputs.add(temp2);
            } else {
                double[][] temp = new double[config[i - 1]][config[i]];
                double[] temp2 = new double[config[i]];
                double[] temp3 = new double[config[i]];
                double[][] temp4 = new double[config[i - 1]][config[i]];
                for (int j = 0; j < temp2.length; j++) {
                    for (int k = 0; k < temp.length; k++) {
                        temp[k][j] = (prng.nextDouble() * 
                                (this.weightRange[1] - this.weightRange[0])) + 
                                this.weightRange[0];
                        temp4[k][j] = 0.0;
                    }
                    temp2[j] = 0.0;
                    temp3[j] = 0.0;
                }
                this.weights.add(temp);
                this.previousWeights.add(temp4);
                this.outputs.add(temp2);
                this.deltas.add(temp3);
            }
        }
    }

    public void train() throws FileNotFoundException {
        for (int round = 0; round < this.rounds; round++) {
            boolean print = false;
            if (this.rounds - round <= this.targetInputs.size()) {
                print = true;
            }
            // Feed-forward block.
            for (int row = 0; row < this.config.length; row++) {
                for (int neuron = 0; neuron < this.outputs.get(row).length; neuron++) {
                    if (row == 0) {
                        double targetInput = this.targetInputs.get(round % this.targetInputs.size())[neuron];
                        this.outputs.get(row)[neuron] = targetInput;
                        if (print) {
                        System.out.println("Target input "
                                + Integer.toString(neuron + 1) + ": "
                                + String.format("%1$.2f", targetInput));
                        }
                    } else if (row > 0) {
                        double sum = 0.0;
                        for (int previousRowNeuron = 0;
                                previousRowNeuron < this.outputs.get(row - 1).length;
                                previousRowNeuron++) {
                            double weight = this.weights.get(row - 1)[previousRowNeuron][neuron];
                            this.previousWeights.get(row - 1)[previousRowNeuron][neuron] = weight;
                            sum += this.outputs.get(row - 1)[previousRowNeuron] * weight;
                        }

                        this.outputs.get(row)[neuron] = sigmoid(sum);
                        if (row == this.config.length - 1 && print) {
                            System.out.println("Trained output "
                                    + Integer.toString(neuron + 1) + ": "
                                    + String.format("%1$.6f", this.outputs.get(row)[neuron]));
                        }
                    }
                }
            }
            //Back-propagation block.
            for (int row = this.config.length - 1; row >= 1; row--) {
                for (int neuron = 0; neuron < this.outputs.get(row).length; neuron++) {
                    if (row == this.config.length - 1) {
                        double output = this.outputs.get(row)[neuron];
                        double targetOutput = this.targetOutputs.get(round % this.targetOutputs.size())[neuron];
                        if (print) {
                            System.out.println("Target output " +
                                    Integer.toString(neuron + 1) + ": "
                                    + String.format("%1$.6f", targetOutput));
                        }
                        double delta = output * (1.0 - output)
                                * (targetOutput - output);
                        this.deltas.get(row - 1)[neuron] = delta;
                        for (int previousRowNeuron = 0;
                                previousRowNeuron < this.outputs.get(row - 1).length;
                                previousRowNeuron++) {
                            this.weights.get(row - 1)[previousRowNeuron][neuron] += this.learningRate
                                    * delta * this.outputs.get(row - 1)[previousRowNeuron];
                        }
                    } else {
                        double output = this.outputs.get(row)[neuron];
                        double error = 0.0;
                        for (int nextRowNeuron = 0;
                                nextRowNeuron < this.outputs.get(row + 1).length;
                                nextRowNeuron++) {
                            error += this.previousWeights.get(row)[neuron][nextRowNeuron]
                                    * this.deltas.get(row)[nextRowNeuron];
                        }
                        double delta = output * (1.0 - output) * error;
                        this.deltas.get(row - 1)[neuron] = delta;
                        for (int previousRowNeuron = 0;
                                previousRowNeuron < this.outputs.get(row - 1).length;
                                previousRowNeuron++) {
                            this.weights.get(row - 1)[previousRowNeuron][neuron] += this.learningRate *
                                    delta * this.outputs.get(row - 1)[previousRowNeuron];
                        }
                    }
                }
            }
            if (print) {
                System.out.println();
            }
            write();
        }
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.pow(Math.E, -x));
    }
    
    private void write() throws FileNotFoundException {
        DataOutputStream fout = new DataOutputStream(
                new FileOutputStream(currentDateInString() + ".ann"));
    }
    
    private String currentDateInString() {
        return dateFormat.format(calendar.getTime());
    }

    private void verify()
            throws ImpossibleNeuralConfigException,
            InvalidTargetInputException,
            InvalidTargetOutputException, 
            ImpossibleWeightRangeException {
        if (this.weightRange.length != 2) {
            throw new ImpossibleWeightRangeException(
                    "The array size of weightRange must be 2");
        }
        
        if (this.weightRange[0] >= this.weightRange[1]) {
            throw new ImpossibleWeightRangeException(
                    "The first element of weightRange must be smaller than the"
                    + "second element.");
        }
        
        if (this.config.length < 2) {
            throw new ImpossibleNeuralConfigException();
        }
        for (double[] d : this.targetInputs) {
            if (this.config[0] != d.length) {
                throw new InvalidTargetInputException();
            }
        }

        for (double[] d : this.targetOutputs) {
            if (this.config[config.length - 1] != d.length) {
                throw new InvalidTargetOutputException();
            }
        }
    }

    private class InvalidTargetInputException extends Exception {

        InvalidTargetInputException() {
            super("Target input must be the same size as stated in config.");
        }
    }

    private class InvalidTargetOutputException extends Exception {

        InvalidTargetOutputException() {
            super("Target output must be the same size as stated in the "
                    + "config.");
        }
    }

    private class ImpossibleNeuralConfigException extends Exception {

        ImpossibleNeuralConfigException() {
            super("Config must have at least 2 array elements, or more");
        }
    }
    
    private class ImpossibleWeightRangeException extends Exception {
        
        ImpossibleWeightRangeException(String msg) {
            super(msg);
        }
    }
}
