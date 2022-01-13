package ca.mattlack.ai.experiments.rein.game.ai;

import ca.mattlack.ai.experiments.rein.game.Render;
import ca.mattlack.ai.experiments.rein.game.WarGame;
import ca.mattlack.ai.experiments.rein.game.WarGameState;
import ca.mattlack.ai.experiments.rein.game.entity.EntityPlayer;
import ca.mattlack.ai.experiments.rein.game.math.Vector2D;
import ca.mattlack.neuralnet.DenseLayer;
import ca.mattlack.neuralnet.Matrix;
import ca.mattlack.neuralnet.Network;
import net.ultragrav.serializer.GravSerializer;

import javax.swing.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PPOTrainer {

    private Network actor = new Network();
    private Network critic = new Network();

    private static SecureRandom secureRandom = new SecureRandom();

    public static void main(String[] args) {
        PPOTrainer trainer = new PPOTrainer();
        boolean training = false;
        if (!training) {
            trainer.run();
        } else {
            trainer.loadNetworks();
            trainer.start();
        }
    }

    public PPOTrainer() {

        // Initialize networks.
        // 57 inputs, 58 outputs. size 128 hidden.
        actor.addLayer(new DenseLayer(57, 20));
        actor.addLayer(new DenseLayer(20, 20));
        actor.addLayer(new DenseLayer(20, 58).setActivationFunctionSigmoid());

        critic.addLayer(new DenseLayer(57, 20));
        critic.addLayer(new DenseLayer(20, 20));
        critic.addLayer(new DenseLayer(20, 1).setActivationFunctionLinear());

        actor.setLearningRate(0.0003);
        critic.setLearningRate(0.0003);
    }

    public void loadNetworks() {
        try {
            actor = loadNetwork(new File("actor.model"));
            critic = loadNetwork(new File("critic.model"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void run() {

        loadNetworks();

        Render render = new Render();
        while (true) {
            WarGame game = new WarGame();
            game.setup();
            int steps = 400;

            PPOPlayer player = new PPOPlayer(actor, critic);
            PPOPlayer player2 = new PPOPlayer(actor, critic);

            double totalReward1 = 0;
            double totalReward2 = 0;
            for (int i = 0; i < steps; i++) {
                WarGameState state = game.getCurrentGameState();
                WarGameState state2 = game.getCurrentGameState();
                double[] actions = player.actAndRecord(state, 0);

                double[] maskedProbs1 = new double[58];

                int[] actionMask1 = new int[58];

                double reward1 = handleActions(game, game.player1, game.flagPos2, actions, maskedProbs1, actionMask1, false);

                double[] actions2 = player2.actAndRecord(state2, 1);

                double[] maskedProbs2 = new double[58];

                int[] actionMask2 = new int[58];

                double reward2 = handleActions(game, game.player2, game.flagPos1, actions2, maskedProbs2, actionMask2, false);

                totalReward1 += reward1;
                totalReward2 += reward2;

                List<Vector2D> highlights = new ArrayList<>();
                if (actionMask1[0] == 1) {
                    int index = 0;
                    for (int j = 0; j < 49; j++) {
                        if (actionMask1[j+1] == 1) {
                            index = j;
                            break;
                        }
                    }
                    int x = index / 7 + (int) game.player1.getPosition().getX() - 3;
                    int y = index % 7 + (int) game.player1.getPosition().getY() - 3;
                    highlights.add(new Vector2D(x, y));
                }
                if (actionMask2[0] == 1) {
                    int index = 0;
                    for (int j = 0; j < 49; j++) {
                        if (actionMask2[j + 1] == 1) {
                            index = j;
                            break;
                        }
                    }
                    int x = index / 7 + (int) game.player2.getPosition().getX() - 3;
                    int y = index % 7 + (int) game.player2.getPosition().getY() - 3;
                    highlights.add(new Vector2D(x, y));
                }

                render.renderFrame(game, highlights.toArray(new Vector2D[0]));

                player.record(reward1, actionMask1, i == steps - 1);
                player2.record(reward2, actionMask2, i == steps - 1);

                // print out last 8 numbers in action mask 1.
                int[] actionMask1Copy = new int[9];
                System.arraycopy(actionMask1, actionMask1.length - 8, actionMask1Copy, 1, 8);
                System.out.println("Player 1 actions: " + Arrays.toString(actionMask1Copy));

                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private volatile boolean run = true;


    public void start() {
        int episodes = 20000;
        double gamma;

        actor.setLearningRate(0.0003);
        critic.setLearningRate(0.0003);

        // Open a window with a single button called stop, that sets run to false.
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JButton button = new JButton("Stop");
        button.addActionListener(e -> run = false);
        frame.add(button);
        frame.pack();
        frame.setVisible(true);

        for (int i = 0; i < episodes && run; i++) {

            gamma = Math.min(episodes * (0.99 / 100), 0.99);
            gamma = 0.00018;

            // Create game instance.
            WarGame game = new WarGame();

            game.setup();

            // Create ppo players.
            PPOPlayer player1 = new PPOPlayer(actor, critic);
            PPOPlayer player2 = new PPOPlayer(actor, critic);

            double totalReward1 = 0;
            double totalReward2 = 0;

            int maxSteps = 1024;
            for (int j = 0; j < maxSteps && game.getWinner() == -1; j++) {

                WarGameState state1 = game.getCurrentGameState();
                WarGameState state2 = game.getCurrentGameState();

                double[] actions1 = player1.actAndRecord(state1, 0);
                double[] actions2 = player2.actAndRecord(state2, 1);

                double[] maskedProbs1 = new double[58];
                double[] maskedProbs2 = new double[58];

                int[] actionMask1 = new int[58];
                int[] actionMask2 = new int[58];

                double reward1 = handleActions(game, game.player1, game.flagPos2, actions1, maskedProbs1, actionMask1, false);
                double reward2 = handleActions(game, game.player2, game.flagPos1, actions2, maskedProbs2, actionMask2, false);

                if (game.getWinner() == 0) {
                    reward1 += 1;
                } else if (game.getWinner() == 1) {
                    reward2 += 1;
                } else if (game.getWinner() == 2) {
                    reward1 += 1;
                    reward2 += 1;
                }

                boolean terminal = game.getWinner() != -1 || j == maxSteps - 1;

                totalReward1 += reward1;
                totalReward2 += reward2;

                player1.record(reward1, actionMask1, terminal);
                player2.record(reward2, actionMask2, terminal);
            }

            player1.collectMemory(player2);
            player1.fit(50, gamma);

            System.out.println("Episode " + i + ": " + totalReward1 + " " + totalReward2 + " WIN: " + game.getWinner());
        }

        try {
            System.out.println("Saving models.");
            saveNetwork(actor, new File("actor.model"));
            saveNetwork(critic, new File("critic.model"));
            System.out.println("Saved models.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double handleActions(WarGame game, EntityPlayer player, Vector2D flagPos, double[] actions, double[] actionsTaken, int[] actionMask, boolean chooseMax) {
        double reward = 0;

        double breakBlock = actions[0];
        if (secureRandom.nextDouble() * 0.005 < breakBlock || chooseMax && breakBlock > 0.5 || flagPos.distance(player.getPosition()) < 5) {
            actionMask[0] = 1;
            actionsTaken[0] = breakBlock;

            double[] dist = new double[49];
            for (int i = 0; i < 49; i++) {
                dist[i] = actions[i + 1];
            }

            int pickedBlock = pickRandom(dist);

            if (chooseMax) {
                double max = 0;
                for (int i = 0; i < 49; i++) {
                    if (dist[i] > max) {
                        max = dist[i];
                        pickedBlock = i;
                    }
                }
            }

            actionMask[pickedBlock + 1] = 1;
            actionsTaken[pickedBlock + 1] = prob(dist, pickedBlock);

            int x = pickedBlock / 7 + (int) player.getPosition().getX() - 3;
            int y = pickedBlock % 7 + (int) player.getPosition().getY() - 3;

            int playerNum = player == game.player1 ? 1 : 2;
            Vector2D flag = playerNum == 1 ? game.flagPos1 : game.flagPos2;
            if (x != (int) flag.getX() || y != (int) flag.getY()) {
                if (game.getWorld().getBlock(x, y) == 1) {
                    game.breakBlock(x, y);
                }
            }
            if (game.getWinner() != -1) {
                System.out.println("Player " + playerNum + " broke a flag.");
            }
        }

        double movX;
        double movY;

        double[] distX = new double[4];
        double[] distY = new double[4];

        double[] choices = {0.6, 0.9, -0.6, -0.9};

        for (int i = 0; i < 4; i++) {
            distX[i] = actions[50 + i];
        }
        for (int i = 0; i < 4; i++) {
            distY[i] = actions[54 + i];
        }

        int xPick = pickRandom(distX);
        int yPick = pickRandom(distY);

        if (chooseMax) {
            double max = 0;
            for (int i = 0; i < 4; i++) {
                if (distX[i] > max) {
                    max = distX[i];
                    xPick = i;
                }
            }
            max = 0;
            for (int i = 0; i < 4; i++) {
                if (distY[i] > max) {
                    max = distY[i];
                    yPick = i;
                }
            }
        }

        actionMask[xPick + 50] = 1;
        actionMask[yPick + 54] = 1;
        actionsTaken[xPick + 50] = prob(distX, xPick);
        actionsTaken[yPick + 54] = prob(distY, yPick);

        movX = choices[xPick];
        movY = choices[yPick];

        Vector2D oldPos = player.getPosition();
        player.move(new Vector2D(movX, movY));
        Vector2D newPos = player.getPosition();

        double distanceDelta = (oldPos.distance(flagPos) - newPos.distance(flagPos));
        reward += 0.25 * distanceDelta;

        return reward;
    }

    public void saveNetwork(Network network, File file) throws IOException {
        GravSerializer serializer = new GravSerializer();

        serializer.writeDouble(network.getLearningRate());

        List<DenseLayer> layerList = network.getLayers();

        serializer.writeInt(layerList.size());

        for (DenseLayer layer : layerList) {
            saveLayer(serializer, layer);
        }

        serializer.writeToStream(new FileOutputStream(file));
    }

    public void saveLayer(GravSerializer serializer, DenseLayer denseLayer) {

        serializer.writeInt(2); // Discriminator
        serializer.writeInt(1); // Version

        Matrix weights = denseLayer.getWeights();
        Matrix biases = denseLayer.getBiases();
        double updateIterations = denseLayer.getUpdateIteration();

        saveMatrix(serializer, weights);
        saveMatrix(serializer, biases);
        saveMatrix(serializer, denseLayer.getM());
        saveMatrix(serializer, denseLayer.getV());
        serializer.writeDouble(updateIterations);

        int activationType = denseLayer.getActivationType();
        serializer.writeInt(activationType);
    }

    public void saveMatrix(GravSerializer serializer, Matrix matrix) {
        int[] dimensions = matrix.getDimensions();
        double[] data = matrix.getData();
        serializer.writeObject(dimensions);
        serializer.writeObject(data);
    }

    public Matrix loadMatrix(GravSerializer serializer) {
        int[] dimensions = serializer.readObject();
        double[] data = serializer.readObject();
        return Matrix.wrap(data, dimensions);
    }

    public DenseLayer loadLayer(GravSerializer serializer) {
        serializer.mark();
        int version = 0;
        int discriminator = serializer.readInt();
        if (discriminator == 2) {
            version = serializer.readInt();
        } else {
            serializer.reset();
        }


        Matrix weights = loadMatrix(serializer);
        Matrix biases = loadMatrix(serializer);

        DenseLayer denseLayer = new DenseLayer(weights.getDimensions()[0], weights.getDimensions()[1]);

        if (version > 0) {
            Matrix m = loadMatrix(serializer);
            Matrix v = loadMatrix(serializer);
            denseLayer.setM(m);
            denseLayer.setV(v);
        }

        double updateIterations = serializer.readDouble();

        int activationType = serializer.readInt();

        denseLayer.setWeights(weights);
        denseLayer.setBiases(biases);
        denseLayer.setUpdateIteration(updateIterations);

        switch (activationType) {
            case 1 -> denseLayer.setActivationFunctionSigmoid();
            case 2 -> denseLayer.setActivationFunctionLinear();
        }

        return denseLayer;
    }

    public Network loadNetwork(File file) throws IOException {
        GravSerializer serializer = new GravSerializer(new FileInputStream(file));

        Network network = new Network();
        network.setLearningRate(serializer.readDouble());

        int layerCount = serializer.readInt();

        for (int i = 0; i < layerCount; i++) {
            network.addLayer(loadLayer(serializer));
        }

        return network;
    }

    public static double[] getActionsTaken(double[] actions, int[] actionMask) {
        double[] actionsTaken = new double[58];
        actionsTaken[0] = actionMask[0] * actions[0];

        double[] dist = toProbs(actions, 1, 50);
        for (int i = 0; i < 49; i++) {
            actionsTaken[i + 1] = actionMask[i + 1] * dist[i];
        }

        dist = toProbs(actions, 50, 54);
        for (int i = 0; i < 4; i++) {
            actionsTaken[i + 50] = actionMask[i + 50] * dist[i];
        }

        dist = toProbs(actions, 54, 58);
        for (int i = 0; i < 4; i++) {
            actionsTaken[i + 54] = actionMask[i + 54] * dist[i];
        }

        return actionsTaken;
    }

    public int pickRandom(double[] distribution) {
        double sum = 0;
        for (double d : distribution) {
            sum += d;
        }

        double rand = secureRandom.nextDouble() * sum;
        double current = 0;
        for (int i = 0; i < distribution.length; i++) {
            current += distribution[i];
            if (rand < current) {
                return i;
            }
        }

        return (int) (secureRandom.nextDouble() * distribution.length);
    }

    /**
     * Gets the probability of an index in the distribution.
     */
    public double prob(double[] distribution, int index) {
        double sum = 0;
        for (double d : distribution) {
            sum += d;
        }

        return distribution[index] / sum;
    }

    /**
     * Convert to a normal distribution.
     */
    public static double[] toProbs(double[] distribution, int startIndex, int endIndex) {
        double sum = 0;
        for (int i = startIndex; i < endIndex; i++) {
            sum += distribution[i];
        }

        double[] probs = new double[endIndex - startIndex];
        for (int i = startIndex; i < endIndex; i++) {
            probs[i - startIndex] = distribution[i] / sum;
        }

        return probs;
    }
}
