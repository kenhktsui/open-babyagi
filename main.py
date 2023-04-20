import time
from collections import deque
from autonomous_agent.memory import Encoder, DenseRetriever
from autonomous_agent.schemas import Task
from autonomous_agent.agents import task_creation_agent, prioritization_agent, execution_agent
from autonomous_agent.logger import TxtLogger
from autonomous_agent.config import Config


class AutonomousAgent:
    def __init__(self, config: Config):
        self.config = config
        self.objective = config.objective
        encoder = Encoder(encoder_name=config.retriever_encoder.model_name)
        self.retriever = DenseRetriever(encoder,
                                        collection_name=config.retriever.collection_name,
                                        qdrant_host=config.retriever.qdrant_host,
                                        qdrant_port=config.retriever.qdrant_port)
        self.logger = TxtLogger(config.logger.log_path)
        self.task_list = deque([])

    def add_task(self, task: Task):
        self.task_list.append(task)

    def run(self):
        # Add the first task
        first_task = Task(task_id=1, task_name=self.config.initial_task)
        self.add_task(first_task)
        # Main loop
        task_id_counter = 1
        loop_idx = 0
        while True:
            self.logger.log("="*30 + f" LOOP {loop_idx} " + "="*30)
            if self.task_list:
                self.logger.log("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
                for t in self.task_list:
                    self.logger.log(str(t.task_id) + ": " + t.task_name)

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                self.logger.log("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
                self.logger.log(str(task.task_id) + ": " + task.task_name)

                # Send to execution function to complete the task based on the context
                result = execution_agent(self.objective, task.task_name, self.config.retriever.top_k,
                                         self.config.llm_client.host, self.config.llm_client.port, self.retriever)
                task.result = result
                task.result_id = f"result_{task.task_id}"
                this_task_id = int(task.task_id)
                self.logger.log("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
                self.logger.log(task.result)
                # Step 2: Enrich result and store in vector DB
                self.retriever.insert(task)
                # Step 3: Create new tasks and reprioritize task list
                new_tasks = task_creation_agent(
                    self.objective,
                    {
                        "data": task.result
                    },
                    task.task_name,
                    [t.task_name for t in self.task_list],
                    self.config.llm_client.host,
                    self.config.llm_client.port
                )

                for new_task in new_tasks:
                    task_id_counter += 1
                    new_task['task_id'] = task_id_counter
                    new_task = Task(**new_task)
                    self.add_task(new_task)
                prioritization_agent(self.objective, self.task_list, this_task_id, self.config.llm_client.host, self.config.llm_client.port)

            self.logger.close()
            time.sleep(5)  # Sleep before checking the task list again
            loop_idx += 1


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser("Autonomous Agent")
    parser.add_argument("config_path", type=str, help="Objective of the agent")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = Config(**yaml.safe_load(f))

    auto_agent = AutonomousAgent(config)
    auto_agent.run()
