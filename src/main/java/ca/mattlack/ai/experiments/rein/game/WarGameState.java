package ca.mattlack.ai.experiments.rein.game;

import ca.mattlack.ai.experiments.rein.game.ai.GameState;
import ca.mattlack.ai.experiments.rein.game.math.Vector2D;

public class WarGameState extends GameState {
    private Vector2D player1Pos;
    private Vector2D player2Pos;

    private int[] player1Env; // 49 ints
    private int[] player2Env;

    private Vector2D player1Flag;
    private Vector2D player2Flag;

    public WarGameState(Vector2D player1Pos, Vector2D player2Pos, int[] player1Env, int[] player2Env, Vector2D player1Flag, Vector2D player2Flag) {
        this.player1Pos = normalizePosition(player1Pos);
        this.player2Pos = normalizePosition(player2Pos);
        this.player1Env = player1Env;
        this.player2Env = player2Env;
        this.player1Flag = normalizePosition(player1Flag);
        this.player2Flag = normalizePosition(player2Flag);
    }


    @Override
    public double[] encoded(int playerNum) {
        // Environment, own position x and y, opponent position x and y, own flag x and y, opponent flag x and y.
        // All x and y positions should be normalized between 0 and 1. (Map is 128x128)

        double[] out = new double[57];
        if (playerNum == 0) {
            for (int i = 0; i < 49; i++) {
                out[i] = player1Env[i];
            }
            out[49] = player1Pos.getX();
            out[50] = player1Pos.getY();
            out[51] = player2Pos.getX();
            out[52] = player2Pos.getY();
            out[53] = player1Flag.getX();
            out[54] = player1Flag.getY();
            out[55] = player2Flag.getX();
            out[56] = player2Flag.getY();
        } else {
            for (int i = 0; i < 49; i++) {
                out[i] = player2Env[i];
            }
            out[49] = player2Pos.getX();
            out[50] = player2Pos.getY();
            out[51] = player1Pos.getX();
            out[52] = player1Pos.getY();
            out[53] = player2Flag.getX();
            out[54] = player2Flag.getY();
            out[55] = player1Flag.getX();
            out[56] = player1Flag.getY();
        }
        return out;
    }

    public Vector2D normalizePosition(Vector2D vector2D) {
        return vector2D.subtract(new Vector2D(64, 64)).divide(64);
    }
}
