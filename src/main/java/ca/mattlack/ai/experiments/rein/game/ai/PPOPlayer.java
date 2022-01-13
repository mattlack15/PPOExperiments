package ca.mattlack.ai.experiments.rein.game.ai;

import ca.mattlack.neuralnet.DenseLayer;
import ca.mattlack.neuralnet.Matrix;
import ca.mattlack.neuralnet.Network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class PPOPlayer {

    private Network actor;
    private Network critic;

    public final List<GameState> gameStateList = new ArrayList<>();
    public final List<int[]> actionList = new ArrayList<>();
    public final List<double[]> probsList = new ArrayList<>();

    public static boolean VERBOSE = false;

    public PPOPlayer(Network actor, Network critic) {
        this.actor = actor;
        this.critic = critic;
    }

    public void setActor(Network actor) {
        this.actor = actor;
    }

    public void setCritic(Network critic) {
        this.critic = critic;
    }

    public Network getActor() {
        return actor;
    }

    public Network getCritic() {
        return critic;
    }

    public double[] actAndRecord(GameState state, int playerNum) {
        double[] encoded = state.encoded(playerNum);
        double[] predictions = actor.propagate(encoded);

        double value = critic.propagate(encoded)[0];
        gameStateList.add(state);
        state.setPlayerNum(playerNum);

        if (probsList.size() == 3 && VERBOSE) {
            // print out the last 8 numbers in predictions.
            double[] lastPredictions = new double[8];
            System.arraycopy(predictions, predictions.length - 8, lastPredictions, 0, 8);
            System.out.println("Predictions: " + Arrays.toString(lastPredictions));
            System.out.println("State Value: " + value);
        }

        state.setValue(value);

        probsList.add(predictions);

        return predictions;
    }

    public void record(double reward, int[] actionsTaken, boolean terminal) {
        if (gameStateList.size() == 0) throw new RuntimeException("No game state to record reward");
        gameStateList.get(gameStateList.size() - 1).setReward(reward);
        gameStateList.get(gameStateList.size() - 1).setTerminal(terminal);
        actionList.add(actionsTaken);
    }

    public List<GameState> getGameStateList() {
        return gameStateList;
    }

    public GameState getLastGameState() {
        return gameStateList.get(gameStateList.size() - 1);
    }

    public void collectMemory(PPOPlayer other) {
        this.gameStateList.addAll(other.gameStateList);
        this.probsList.addAll(other.probsList);
        this.actionList.addAll(other.actionList);
        other.gameStateList.clear();
        other.probsList.clear();
        other.actionList.clear();
    }

    /**
     * Computes the generalized advantage estimate.
     *
     * @param gamma  discount factor
     * @param lambda smoothing parameter, usually 0.95
     */
    public double[] advantages(double gamma, double lambda) {
        double[] advantages = new double[gameStateList.size()];
        double[] rewards = new double[gameStateList.size()];
        double[] values = new double[gameStateList.size()];

        for (int i = 0; i < gameStateList.size(); i++) {
            rewards[i] = gameStateList.get(i).getReward();
            values[i] = gameStateList.get(i).getValue();
        }

//        for (int i = gameStateList.size() - 1; i >= 0; i--) {
//            double delta = rewards[i] - values[i];
//            if (gameStateList.get(i).isTerminal()) {
//                // Non-terminal state
//                delta += gamma * values[i + 1];
//                advantages[i] = delta + gamma * lambda * advantages[i + 1];
//            } else {
//                // Terminal state
//                advantages[i] = delta;
//            }
//        }

        // For now just use discounted rewards.
        double[] discountedRewards = new double[rewards.length];
        for (int i = discountedRewards.length - 1; i >= 0; i--) {
            if (gameStateList.get(i).isTerminal()) {
                discountedRewards[i] = rewards[i];
            } else {
                discountedRewards[i] = rewards[i] + gamma * discountedRewards[i + 1];
            }
            advantages[i] = discountedRewards[i] - values[i];
        }
        return (advantages);
    }

    public void fit(int batchSize, double gamma) {
        if (gameStateList.size() == 0) throw new IllegalStateException("No game states to fit to");

        double clipParam = 0.1;

        double[] advantages = advantages(gamma, 0.95);
        GameState[] states = gameStateList.toArray(new GameState[0]);
        double[] values = new double[states.length];
        for (int i = 0; i < states.length; i++) values[i] = states[i].value;
        int[][] actions = new int[states.length][];
        for (int i = 0; i < states.length; i++) actions[i] = this.actionList.get(i);
        double[][] probs = new double[states.length][];
        for (int i = 0; i < states.length; i++) probs[i] = this.probsList.get(i);

        int epochs = 1;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Split into batches.
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < gameStateList.size(); i += batchSize) {
                indices.add(i);
            }

            // Shuffle.
            Collections.shuffle(indices);

            int[] batches = indices.stream().mapToInt(i -> i).toArray();

            // Go through the batches.
            for (int i = 0; i < batches.length; i++) {
                int start = batches[i];
                int end = Math.min(start + batchSize, gameStateList.size());

                GameState[] batchStates = new GameState[end - start];
                int[][] batchActions = new int[end - start][]; // one-hot encoded (ex. [0,0,0,1,0,0])
                double[] batchAdvantages = new double[end - start];
                double[] batchValues = new double[end - start];
                double[][] batchProbs = new double[end - start][];

                System.arraycopy(states, start, batchStates, 0, end - start);
                System.arraycopy(actions, start, batchActions, 0, end - start);
                System.arraycopy(advantages, start, batchAdvantages, 0, end - start);
                System.arraycopy(values, start, batchValues, 0, end - start);
                System.arraycopy(probs, start, batchProbs, 0, end - start);

                // The old probabilities are the probs.
                double[][] oldProbs = batchProbs;

                // Produce the new probabilities.
                double[][] newProbs = new double[batchStates.length][];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    newProbs[i1] = PPOTrainer.getActionsTaken(actor.propagate(batchStates[i1].encoded()), batchActions[i1]);

                }

                // Mask the old probs.
                for (int i1 = 0; i1 < oldProbs.length; i1++) {
                    oldProbs[i1] = PPOTrainer.getActionsTaken(oldProbs[i1], batchActions[i1]);
                }

                // The ratios are the ratios between the new probabilities and the old ones.
                double[][] ratios = new double[batchStates.length][];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    ratios[i1] = new double[newProbs[i1].length];
                    for (int i2 = 0; i2 < newProbs[i1].length; i2++) {
                        ratios[i1][i2] = Math.exp(Math.log(newProbs[i1][i2] + 1E-10) - Math.log(oldProbs[i1][i2] + 1E-10)); // Law of exponents, this is the same as division.
                    }
                }

                // Weighted with the advantages.
                double[][] weightedRatios = new double[batchStates.length][];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    weightedRatios[i1] = new double[newProbs[i1].length];
                    for (int i2 = 0; i2 < newProbs[i1].length; i2++) {
                        weightedRatios[i1][i2] = ratios[i1][i2] * batchAdvantages[i1];
                    }
                }

                // Clipped between 1-clip and 1+clip.
                double[][] clippedRatios = new double[batchStates.length][];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    clippedRatios[i1] = new double[newProbs[i1].length];
                    for (int i2 = 0; i2 < newProbs[i1].length; i2++) {
                        clippedRatios[i1][i2] = Math.min(Math.max(weightedRatios[i1][i2], 1 - clipParam), 1 + clipParam);
                        clippedRatios[i1][i2] *= batchAdvantages[i1];
                    }
                }

//                // Actor loss is minimum between clipped ratios and un-clipped ratios.
//                double[][] actorLosses = new double[batchStates.length][];
//                for (int i1 = 0; i1 < batchStates.length; i1++) {
//                    actorLosses[i1] = new double[newProbs[i1].length];
//                    for (int i2 = 0; i2 < newProbs[i1].length; i2++) {
//                        actorLosses[i1][i2] = Math.min(clippedRatios[i1][i2], weightedRatios[i1][i2]);
//                    }
//                }

//                // Now it's the mean of that.
//                double actorLoss = 0;
//                for (int i1 = 0; i1 < batchStates.length; i1++) {
//                    for (int i2 = 0; i2 < newProbs[i1].length; i2++) {
//                        actorLoss += actorLosses[i1][i2];
//                    }
//                }
//                actorLoss /= batchStates.length;

                // Critic loss.
                double[] criticValues = new double[batchStates.length];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    criticValues[i1] = critic.propagate(batchStates[i1].encoded())[0];
                }
//
//                double criticLoss = 0;
//                for (int i1 = 0; i1 < batchStates.length; i1++) {
//                    criticLoss += Math.pow((batchAdvantages[i1] + batchValues[i1]) - criticValues[i1], 2);
//                }
//                criticLoss /= batchStates.length;
//
////            System.out.println("Critic Loss: " + criticLoss);
//
//                double totalLoss = actorLoss + (criticLoss * 0.5);

//            System.out.println("Total Loss: " + totalLoss);
//            System.out.println("Actor Loss: " + actorLoss + " Critic Loss: " + criticLoss);

                // Compute gradient. It is advantage / (numActions * oldProbs)
                double[][] actorGradient = new double[batchStates.length][];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    actorGradient[i1] = new double[newProbs[i1].length];
                    for (int i2 = 0; i2 < newProbs[i1].length; i2++) {
                        // Possibly remove newProbs[i1].length from this as it might slow down convergence.
                        if (batchActions[i1][i2] == 0) {
                            actorGradient[i1][i2] = 0;
                        } else {
                            actorGradient[i1][i2] = batchAdvantages[i1] / (oldProbs[i1][i2] + 1E-10);
                            if (clippedRatios[i1][i2] < weightedRatios[i1][i2]) { // If the clipped value was used.
                                double ratio = ratios[i1][i2];
                                if (ratio > 1.0 + clipParam || ratio < 1.0 - clipParam) { // And it was clipped
                                    actorGradient[i1][i2] = 0; // Don't update any further, gradient is zero.
                                }
                            }
                        }
                    }
                    if (i1 == 0 && VERBOSE) {
                        // Print out the last 8 numbers in the gradient, and the batchActions, and the advantages.
                        double[] lastNumsGradient = new double[8];
                        double[] lastNumsActions = new double[8];
                        for (int i2 = 7; i2 >= 0; i2--) {
                            lastNumsGradient[7 - i2] = actorGradient[i1][newProbs[i1].length - 1 - i2];
                            lastNumsActions[7 - i2] = batchActions[i1][newProbs[i1].length - 1 - i2];
                        }
                        System.out.println("Gradient: " + Arrays.toString(lastNumsGradient) + "\nActions: " + Arrays.toString(lastNumsActions) + "\nAdvantage: " + batchAdvantages[i1]);
                        System.out.println("Reward at this timestep: " + batchStates[i1].reward);
                        System.out.println("Value at this timestep: " + criticValues[i1] + "\n");
                        System.out.println("Ratios at this timestep: " + Arrays.toString(ratios[i1]) + "\n");
                    }
                }

                // Back propagate for actor.
                for (double[] grad : actorGradient) {
                    Matrix gradientMatrix = Matrix.wrap(grad);
                    for (int i2 = actor.getLayers().size() - 1; i2 >= 0; i2--) {
                        DenseLayer layer = actor.getLayers().get(i2);
                        gradientMatrix = layer.backPropagate(gradientMatrix);
                    }
                }
                actor.getLayers().forEach(layer -> layer.updateParams(actor.getLearningRate()));

                // Back propagate for critic.
                double[] criticGradient = new double[batchStates.length];
                for (int i1 = 0; i1 < batchStates.length; i1++) {
                    criticGradient[i1] = (batchAdvantages[i1] + batchValues[i1]) - criticValues[i1];
                }

                for (double grad : criticGradient) {
                    Matrix gradientMatrix = Matrix.wrap(new double[]{grad});
                    for (int i2 = critic.getLayers().size() - 1; i2 >= 0; i2--) {
                        DenseLayer layer = critic.getLayers().get(i2);
                        gradientMatrix = layer.backPropagate(gradientMatrix);
                    }
                }
                critic.getLayers().forEach(layer -> layer.updateParams(critic.getLearningRate()));

            }
        }

        // Clear memory.
        gameStateList.clear();
        actionList.clear();
        probsList.clear();
    }

    public static double[] standardize(double[] arr) {
        double[] standardized = new double[arr.length];
        double mean = 0;
        double std = 0;
        for (int i1 = 0; i1 < arr.length; i1++) {
            mean += arr[i1];
        }
        mean /= arr.length;
        for (int i1 = 0; i1 < arr.length; i1++) {
            std += Math.pow(arr[i1] - mean, 2);
        }
        std /= arr.length;
        std = Math.sqrt(std);
        std += 1e-10;
        for (int i1 = 0; i1 < arr.length; i1++) {
            standardized[i1] = (arr[i1] - mean) / std;
        }
        return standardized;
    }
}
