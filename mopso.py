class Particle:
    def __init__(self):
        self.position = np.array([np.random.uniform(0.25, 0.5), np.random.uniform(0, 0.25), np.random.uniform(0.25, 0.5), np.random.uniform(0, 0.25),
                                  np.random.uniform(0.25, 0.65), np.random.uniform(0, 0.25), np.random.uniform(0.25, 0.65), np.random.uniform(0, 0.25)]) # 8次元
        self.velocity = np.random.rand(8)
        self.best_position = np.copy(self.position) 
        self.best_score = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]

class MOPSO:
    def __init__(self, num_particles, num_iterations, model):
        self.num_particles = num_particles
        self.num_iterations = num_iterations 
        self.particles = [Particle() for _ in range(num_particles)]
        self.gbest_position = np.array([np.random.uniform(0.25, 0.5), np.random.uniform(0, 0.25), np.random.uniform(0.25, 0.5), np.random.uniform(0, 0.25),
                                        np.random.uniform(0.25, 0.65), np.random.uniform(0, 0.25), np.random.uniform(0.25, 0.65), np.random.uniform(0, 0.25)
                                        ])
        self.gbest_positions_history = []
        self.gbest_score = [float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf')]
        self.gbest_scores_history = [] 
        self.all_scores = []
        self.all_positions = []
        self.model = model  # 学習済みモデル

    def optimize(self):
        for _ in range(self.num_iterations):
            for particle in self.particles:
                objectives = self.evaluate(particle.position)
              
                if self.is_better(objectives, particle.best_score):
                    particle.best_score = objectives
                    particle.best_position = particle.position.copy()

                if self.is_better(objectives, self.gbest_score):
                    self.gbest_score = objectives
                    self.gbest_position = particle.position.copy()
                    self.gbest_positions_history.append(self.gbest_position.copy())
                    self.gbest_scores_history.append(self.gbest_score.copy())

            for particle in self.particles:
                inertia = 0.9
                personal_attraction = 0.5
                global_attraction = 0.9
                r1 = np.random.rand(8)
                r2 = np.random.rand(8)
                cognitive = personal_attraction * r1 * (particle.best_position - particle.position) 
                social = global_attraction * r2 * (self.gbest_position - particle.position)
                particle.velocity = inertia * particle.velocity + cognitive + social
                particle.position += particle.velocity

                # パラメータ範囲の制約を適用
                particle.position[0] = np.clip(particle.position[0], 0.3, 0.5)
                particle.position[1] = np.clip(particle.position[1], 0, 0.25)
                particle.position[2] = np.clip(particle.position[2], 0.3, 0.5)
                particle.position[3] = np.clip(particle.position[3], 0, 0.25)
                particle.position[4] = np.clip(particle.position[4], 0.3, 0.5)
                particle.position[5] = np.clip(particle.position[5], 0, 0.25)
                particle.position[6] = np.clip(particle.position[6], 0.3, 0.5)
                particle.position[7] = np.clip(particle.position[7], 0, 0.25)

                self.all_scores.append(particle.best_score)
                self.all_positions.append(particle.best_position)

        top_scores = sorted(zip(self.all_scores, self.all_positions), key=lambda x: sum(x[0]))
        return top_scores[:30]

    def evaluate(self, position):
        inputs = np.array([[position]]).reshape(1, -1)
        predicted = self.model(inputs)
        obj1 = np.square(predicted[0][0].numpy() - target_w11_max).item()
        obj2 = np.square(predicted[0][1].numpy() - target_BandGap1).item()
        return [obj1, obj2]

    def is_better(self, objectives, best_objectives):
        return sum(objectives) < sum(best_objectives)

mopso = MOPSO(num_particles=30, num_iterations=100, model=forward_model)
top_30_results = mopso.optimize()

for i, (score, position) in enumerate(top_30_results, 1):
    formatted_position = [f"{p:.4f}" for p in position] 
    #formatted_score = [f"{s:.4f}" for s in score]   
    print(f"Position #{i}: {formatted_position}, Score: {score}")
