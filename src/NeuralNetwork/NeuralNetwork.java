package NeuralNetwork;

import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;

class NeuralNetwork {

	private static SimpleDateFormat dateFormat = 
			new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss");
	private static SecureRandom prng = new SecureRandom();
	// Used for Nguyen-Widrow weight initialization.
	// http://www.heatonresearch.com/encog/articles/nguyen-widrow-neural-network-weight.html
	private double beta;
	private double learningRate;
	private double tolerance;
	private int[] config;
	private int[] weightRange;
	private ArrayList<double[]> targetInputSets;
	private ArrayList<double[]> targetOutputSets;
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
			double tolerance)
					throws ImpossibleNeuralConfigException,
					InvalidTargetOutputException,
					InvalidTargetInputException, 
					ImpossibleWeightRangeException {
		// Setting parameters in object for later access.
		this.config = config;
				this.targetInputSets = targetInput;
				this.targetOutputSets = targetOutput;
				this.learningRate = learningRate;
				this.tolerance = tolerance;
				this.weightRange = weightRange;
				this.verify();
				double h = 0.0;
				double numberOfInputs = (double) config[0];
				for (int j = 1; j < config.length - 1; ++j) {
					h += (double) config[j];
				}
				// Nguyen-Widrow weight initialization.
				// (1) :
				//				   1
				//				   -
				// beta  =  0.7 (h)i
				this.beta = 0.7 * (Math.pow(h, 1.0/numberOfInputs));
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
						for (int j = 0; j < temp2.length; j++) {
							//
							// (2) :
							// 
							//        (i<w           )
							//        (   max     2  )
							// n =SQRT(S U M (   w  ))
							//        (i=0        i  )
							//
							 double n = 0.0;
							 for (int k = 0; k < temp.length; k++) {
								 
								 // SUM(...)
								 double d = temp[k][j] * temp[k][j];
								 n += d;
							 }	 
							 // SQRT(...)
							 n = Math.sqrt(n);
							 //
							 // (3) :
							 //       beta w
							 // 	        t+1
							 // w   = ---------
							 //  t+1      n
							 //
							 for (int k = 0; k < temp.length; k++) {
								 temp[k][j] = (temp[k][j] * beta) / (n);
							 }
						}
						this.weights.add(temp);
						this.previousWeights.add(temp4);
						this.outputs.add(temp2);
						this.deltas.add(temp3);
					}
				}
	}

	public void train() throws FileNotFoundException {
		System.out.println("Training started at: " + currentDateInString());
		long start = System.currentTimeMillis();
		int round = 0;
		boolean continueLooping = true;
		while (continueLooping) {
			if (round == this.targetInputSets.size()) {
				round = 0;
			}
			feedForwardLoop(this.targetInputSets.get(round));
			continueLooping = !backPropagationLoop(this.targetOutputSets.get(round));
			round++;
		}
		long elapsed = System.currentTimeMillis() - start;
		System.out.println("Training finished at: " + currentDateInString());
		System.out.println("Training took " + Float.toString(elapsed/1000F) + " seconds.");
		System.out.println();
		printResults();
	}
	
	private void feedForwardLoop(double[] targetInputSet) {
		for (int row = 0; row < this.config.length; row++) {
			for (int neuron = 0; neuron < this.outputs.get(row).length; neuron++) {
				if (row == 0) {
					double targetInput = targetInputSet[neuron];
					this.outputs.get(0)[neuron] = targetInput;
				} else if (row > 0) {
					double sum = 0.0;
					for (int previousRowNeuron = 0;
							previousRowNeuron < this.outputs.get(row - 1).length;
							previousRowNeuron++) {
						double weight = this.weights.get(row - 1)[previousRowNeuron][neuron];
						this.previousWeights.get(row - 1)[previousRowNeuron][neuron] = weight;
						sum += this.outputs.get(row - 1)[previousRowNeuron] * weight;
					}

					this.outputs.get(row)[neuron] = elliott(sum, 1.0, false);
				}
			}
		}
	}
	
	private boolean backPropagationLoop(double[] targetOutputSet) {
		double biggestErrorRate = 0.0;
		// The back-propagation loop.
		for (int row = this.config.length - 1; row >= 1; row--) {
			for (int neuron = 0; neuron < this.outputs.get(row).length; neuron++) {
				if (row == this.config.length - 1) {
					double output = this.outputs.get(row)[neuron];
					double errorRate = targetOutputSet[neuron] - output;
					double absErrorRate = Math.abs(errorRate);
					if (biggestErrorRate < absErrorRate) {
						biggestErrorRate = absErrorRate;
					}
 					double delta = output * (1.0 - output)
							* (errorRate);
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
		if (biggestErrorRate > this.tolerance) {
			return false;
		} else {
			return true;
		}
	}

	private static double elliott(double x, double s, boolean isSymmetric) {
		return (isSymmetric) ? ((x * s) / (1.0 + Math.abs(x * s))) :
			((0.5*(x*s) / (1 + Math.abs(x * s))) + 0.5);
	}

	private void write() throws FileNotFoundException {
		DataOutputStream fout = new DataOutputStream(
				new FileOutputStream(currentDateInString() + ".ann"));
		try {
			fout.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private String currentDateInString() {
		return dateFormat.format(Calendar.getInstance().getTime());
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
		for (double[] d : this.targetInputSets) {
			if (this.config[0] != d.length) {
				throw new InvalidTargetInputException();
			}
		}

		for (double[] d : this.targetOutputSets) {
			if (this.config[config.length - 1] != d.length) {
				throw new InvalidTargetOutputException();
			}
		}
	}
	
	private void printResults() {
		System.out.println("RESULTS: ");
		for (int i = 0; i < this.targetInputSets.size(); ++i) {
			double[] targetInputSet = this.targetInputSets.get(i);
			double[] targetOutputSet = this.targetOutputSets.get(i);
			feedForwardLoop(targetInputSet);
			double[] calculatedOutputSet = this.outputs.get(this.outputs.size() - 1);
			for (int j = 0; j < targetInputSet.length; ++j) {
				System.out.println("Target Input: " + Double.toString(targetInputSet[j]));
			}
			
			for (int j = 0; j < targetOutputSet.length; ++j) {
				System.out.println("Target Output: " + Double.toString(targetOutputSet[j]));
				System.out.println("Calculated output: " + Double.toString(calculatedOutputSet[j]));
			}
			System.out.println();
		}
	}

	private class InvalidTargetInputException extends Exception {

		/**
		 * 
		 */
		private static final long serialVersionUID = 8018808994851307099L;

		InvalidTargetInputException() {
			super("Target input must be the same size as stated in config.");
		}
	}

	private class InvalidTargetOutputException extends Exception {

		private static final long serialVersionUID = -5915356528510734074L;

		InvalidTargetOutputException() {
			super("Target output must be the same size as stated in the "
					+ "config.");
		}
	}

	private class ImpossibleNeuralConfigException extends Exception {

		private static final long serialVersionUID = -1111226417680071610L;

		ImpossibleNeuralConfigException() {
			super("Config must have at least 2 array elements, or more");
		}
	}

	private class ImpossibleWeightRangeException extends Exception {

		private static final long serialVersionUID = -8358484972565353377L;

		ImpossibleWeightRangeException(String msg) {
			super(msg);
		}
	}
}
