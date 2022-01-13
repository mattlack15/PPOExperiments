package ca.mattlack.ai.experiments.rein.game.ai;

public abstract class GameState {

    public double value;
    public double reward;
    public boolean terminal;
    public int playerNum;

    public void setReward(double reward) {
        this.reward = reward;
    }

    public int getPlayerNum() {
        return playerNum;
    }

    public void setPlayerNum(int playerNum) {
        this.playerNum = playerNum;
    }

    public double getReward() {
        return reward;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public abstract double[] encoded(int playerNum);

    public double[] encoded() {
        return encoded(playerNum);
    }

    public void setTerminal(boolean terminal) {
        this.terminal = terminal;
    }

    public boolean isTerminal() {
        return terminal;
    }
}
