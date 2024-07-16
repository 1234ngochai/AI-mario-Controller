package agents.EAController;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.GameStatus;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Agent implements MarioAgent {

    private static final int PLANNING_HORIZON = 40; // Plan ahead for 30 ticks
    private static final int SAFETY_FRAMES = 20; // Additional frames to evaluate safety
    private static final int TOTAL_HORIZON = PLANNING_HORIZON + SAFETY_FRAMES; // Total frames to simulate
    private static final int NUM_ACTION_SEQUENCES = 100; // Number of action sequences to evaluate
    private static final int TOP_SELECTION_SIZE = 10; // Number of top sequences to select
    private static final double MUTATION_RATE = 0.2; // Probability of mutation
    private static final int NUM_GENERATIONS = 30; // Number of generations to evolve 10 as the default
    private static final int STUCK_FRAMES_THRESHOLD = 5; // Number of frames to detect being stuck
    private static final float STUCK_POSITION_THRESHOLD = 5.0f; // Position change threshold to detect being stuck

    private Random random;
    private boolean[][] bestActionSequence; // Store the best action sequence
    private int currentTickInPlan; // Track the current tick in the stored plan



    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        random = new Random();
        bestActionSequence = null;
        currentTickInPlan = 0;
    }

    @Override
    public void train(MarioForwardModel model) {
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        if (model.getCompletionPercentage() > 0.99 || model.getGameStatus() == GameStatus.WIN) {
            // If the game status is WIN, ensure Mario continues moving right
            return new boolean[]{false, true, false, false, false};
        }

        // Check if we need to recalculate the plan
        if (bestActionSequence == null || currentTickInPlan >= PLANNING_HORIZON) {
            bestActionSequence = calculateBestActionSequence(model);
            currentTickInPlan = 0;
        }

        // Execute the next action in the best sequence
        boolean[] action = bestActionSequence[currentTickInPlan];
        currentTickInPlan++;

        return action;
    }

    @Override
    public String getAgentName() {
        return "EvolutionaryMarioAgent";
    }

    private boolean[][] calculateBestActionSequence(MarioForwardModel model) {
        List<boolean[][]> population = new ArrayList<>();
        for (int i = 0; i < NUM_ACTION_SEQUENCES; i++) {
            population.add(randomActionSequence());
        }

        for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
            List<Double> fitnessScores = new ArrayList<>();
            for (boolean[][] actionSequence : population) {
                MarioForwardModel cloneModel = model.clone();
                double fitness = evaluateFitness(actionSequence, cloneModel);
                fitnessScores.add(fitness);
            }

            // Select the top-performing sequences
            List<boolean[][]> topActionSequences = new ArrayList<>();
            for (int i = 0; i < TOP_SELECTION_SIZE; i++) {
                int bestIndex = selectBest(fitnessScores);
                topActionSequences.add(population.get(bestIndex));
                fitnessScores.set(bestIndex, Double.NEGATIVE_INFINITY); // Mark as selected
            }

            // Breed new sequences from the top-performing sequences
            List<boolean[][]> newPopulation = new ArrayList<>(topActionSequences);
            while (newPopulation.size() < NUM_ACTION_SEQUENCES) {
                boolean[][] parent1 = topActionSequences.get(random.nextInt(TOP_SELECTION_SIZE));
                boolean[][] parent2 = topActionSequences.get(random.nextInt(TOP_SELECTION_SIZE));
                boolean[][] child = breed(parent1, parent2);
                mutate(child);
                newPopulation.add(child);
            }

            // Update the population
            population = newPopulation;
        }

        // Evaluate the final population to select the best action sequence
        double bestFitness = Double.NEGATIVE_INFINITY;
        boolean[][] bestActionSequence = new boolean[PLANNING_HORIZON][5];
        for (boolean[][] actionSequence : population) {
            MarioForwardModel cloneModel = model.clone();
            double fitness = evaluateFitness(actionSequence, cloneModel);
            if (fitness > bestFitness) {
                bestFitness = fitness;
                bestActionSequence = actionSequence; // Choose the best sequence
            }
        }

        return bestActionSequence;
    }

    private boolean[][] randomActionSequence() {
        boolean[][] actionSequence = new boolean[TOTAL_HORIZON][5];
        for (int i = 0; i < TOTAL_HORIZON; i++) {
            for (int j = 0; j < 5; j++) {
                actionSequence[i][j] = random.nextBoolean();
            }
        }
        return actionSequence;
    }
    private double evaluateFitness(boolean[][] actionSequence, MarioForwardModel model) {
        double score = 0.0;
        List<Float> positions = new ArrayList<>();
        double totalYPosition = 0;
        int countYPositions = 0;

        for (int tick = 0; tick < TOTAL_HORIZON; tick++) {
            boolean[] actions = actionSequence[tick];
            model.advance(actions);

            float currentPositionX = model.getMarioFloatPos()[0];
            positions.add(currentPositionX);

            // Accumulate Y position and count the number of frames
            float currentYPosition = model.getMarioFloatPos()[1];
            totalYPosition += currentYPosition;
            countYPositions++;

            if (tick == PLANNING_HORIZON && model.getGameStatus() == GameStatus.LOSE) {
                score -= 1000000; // Penalty for losing at the planning horizon
            }

            if (model.getGameStatus() == GameStatus.WIN) {
                score += 10000; // Large bonus for winning
                break;
            } else if (model.getGameStatus() == GameStatus.LOSE) {
                score -= 1000000; // Moderate penalty for losing
                break;
            }
        }

        // Add completion percentage to the score
        score += model.getCompletionPercentage() * 100; // Weight completion percentage heavily

        // Calculate the average Y position
        double averageYPosition = totalYPosition / countYPositions;
        // Calculate the score from the average Y position, scaled to be between 0.1% of the completion score
        double maxReward = score * 0.001; // Maximum reward is 10% of the completion score
        double minYPosition = 0; // Assuming ground level is 0, adjust as necessary
        double maxYPositionScale = 100; // Adjust this scale based on the game level's max height
        double yPositionFactor = (averageYPosition - minYPosition) / maxYPositionScale;
        double yPositionScore = yPositionFactor * maxReward;


        // Total score is the completion score plus the scaled Y position score
        score += yPositionScore;

        return score;
    }

    private boolean isLingering(List<Float> positions) {
        if (positions.size() < STUCK_FRAMES_THRESHOLD) {
            return false;
        }

        float minPosition = Float.MAX_VALUE;
        float maxPosition = Float.MIN_VALUE;

        for (float pos : positions) {
            if (pos < minPosition) {
                minPosition = pos;
            }
            if (pos > maxPosition) {
                maxPosition = pos;
            }
        }

        return (maxPosition - minPosition) < STUCK_POSITION_THRESHOLD;
    }



    private int selectBest(List<Double> fitnessScores) {
        double bestFitness = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;
        for (int i = 0; i < fitnessScores.size(); i++) {
            if (fitnessScores.get(i) > bestFitness) {
                bestFitness = fitnessScores.get(i);
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private boolean[][] breed(boolean[][] parent1, boolean[][] parent2) {
        boolean[][] child = new boolean[TOTAL_HORIZON][5];
        for (int i = 0; i < TOTAL_HORIZON; i++) {
            for (int j = 0; j < 5; j++) {
                child[i][j] = random.nextBoolean() ? parent1[i][j] : parent2[i][j];
            }
        }
        return child;
    }

    private void mutate(boolean[][] genotype) {
        for (int i = 0; i < TOTAL_HORIZON; i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                int action = random.nextInt(5);
                genotype[i][action] = !genotype[i][action];
            }
        }
    }
}
