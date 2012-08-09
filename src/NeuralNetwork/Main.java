package NeuralNetwork;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        ArrayList<double[]> targetInput = new ArrayList<>();
        double[] targetInput1 = {0.0, 0.0};
        double[] targetInput2 = {0.0, 1.0};
        double[] targetInput3 = {1.0, 0.0};
        double[] targetInput4 = {1.0, 1.0};
        targetInput.add(targetInput1);
        targetInput.add(targetInput2);
        targetInput.add(targetInput3);
        targetInput.add(targetInput4);

        ArrayList<double[]> targetOutput = new ArrayList<>();
        double[] targetOutput1 = {0.0};
        double[] targetOutput2 = {1.0};
        double[] targetOutput3 = {1.0};
        double[] targetOutput4 = {0.0};
        targetOutput.add(targetOutput1);
        targetOutput.add(targetOutput2);
        targetOutput.add(targetOutput3);
        targetOutput.add(targetOutput4);
        
        int[] config = {2, 4, 4, 4, 4, 1};
        int[] weightRange = {-2, 2};
        long rounds = 100000;
        double learningRate = 0.3;
        
        try {
            NeuralNetwork nn = new NeuralNetwork(
                    config, targetInput, targetOutput, weightRange, learningRate, rounds);
            nn.train();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
