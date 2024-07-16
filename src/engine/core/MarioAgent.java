package engine.core;

/**
 * Interface for agents that want to play in the framework
 *
 * @author AhmedKhalifa
 */
public interface MarioAgent {

    /**
     * initialize and prepare the agent before the game starts
     *
     * @param model a forward model object so the agent can simulate or initialize some parameters based on it.
     * @param timer amount of time before the agent should return to ensure timely frames.
     */
    void initialize(MarioForwardModel model, MarioTimer timer);

    /**
     * Use this to train your model any way you would like after initialise is called, but before getActions is called the first time
     *
     * @param model a forward model object so the agent can simulate or initialize some parameters based on it.
     */
    void train(MarioForwardModel model);

    /**
     * get mario current actions
     *
     * @param model a forward model object so the agent can simulate the future.
     * @param timer amount of time before the agent has to return the actions to ensure timely frames.
     * @return an array of the state of the buttons on the controller
     */
    boolean[] getActions(MarioForwardModel model, MarioTimer timer);

    /**
     * Return the name of the agent that will be displayed in debug purposes
     *
     * @return
     */
    String getAgentName();
}
