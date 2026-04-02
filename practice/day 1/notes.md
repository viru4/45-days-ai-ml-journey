📘 Day 1 (RL Basics) — Summary Notes
🔷 1. Reinforcement Learning (RL)
Learning by interaction with environment
No labeled data
Based on reward & punishment
Core Loop:
State → Action → Reward → Next State → Repeat
🔷 2. RL Components
Term           Meaning
Agent          Learner
Environment    World
State (S)      Current situation
Action (A)     Decision
Reward (R)     Feedback
Policy (π)     Strategy
🔷 3. RL vs Supervised Learning
RL               Supervised
No labels        Labeled data
Trial & error    Direct learning
Delayed reward   Immediate feedback

🔷 4. Markov Decision Process (MDP)

Defined as:

(S, A, P, R, γ)
S → States
A → Actions
P → Transition probability
R → Reward
γ → Discount factor
🔥 Markov Property:

Future depends only on current state (not past)

🔷 5. Bellman Equation
Value Function:
V(s) = R + γV(s')
Q Function:
Q(s,a) = R + γ max Q(s',a')

👉 Recursive → foundation of RL

🔷 6. Policy vs Value-Based
Value-Based:
Learn Q(s,a)
Use argmax
Policy-Based:
Learn π(a|s)
Output probabilities
🔷 7. Exploration vs Exploitation
Exploration → try new actions
Exploitation → use best known action
🔷 8. Epsilon-Greedy
With ε → random action
With (1-ε) → best action
Epsilon Decay:
Start high → explore
End low → exploit
🔷 9. Episode
One complete run from start → end
Ends when:
Goal reached
Failure
Time limit
🔷 10. Gymnasium + CartPole
Environment: "CartPole-v1"
Action:
0 → left
1 → right
Reward: +1 per step