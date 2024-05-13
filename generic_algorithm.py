import random
from typing import Self

# Homework: A Prototype of Generic Algorithm by QI Guangyao

class Chromosome:
    """Chromomsome that is used for generating inputed strings
    In the homework, assume the lengh of inputed string is no longer than 20.
    """
    # Basic Class Parameters
    gene_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.!? "
    mutation_rate = 0.5
    target = "Hello, World!"
    length = len(target)

    def __init__(self, genes: list):
        """Constructor
        Don't forget to update_fitness() after initialization
        """
        # if genes is empty, randomly initialize it
        # else just let self.genes = genes
        if genes == []:
            for i in range(self.length):
                genes.append(self.gene_pool[random.randrange(0, len(self.gene_pool))])
        self.genes = genes
        # calculate the fitness value by counting how many chars are right
        self.fitness = 0
    
    def __mutate(self) -> None:
        """randomly mutate
        Returns:
            _type_: None
        """
        if random.uniform(0, 1) > self.mutation_rate:
            return
        index = random.randrange(0, self.length)
        self.genes[index] = self.gene_pool[random.randrange(0, len(self.gene_pool))]

    def to_string(self) -> str:
        """return the string of genes of the chromosome

        Returns:
            str: the string format of genes of the chromosome
        """
        return ''.join(self.genes)
    
    def set_genes(self, new_genes: list) -> None:
        """replace genes with a new_genes

        Args:
            new_genes (list): the list of new genes
        """
        self.genes = new_genes
    
    def set_genes_from_index(self, index: int, s:str) -> None:
        """set the genes at given index

        Args:
            index (int): given index
            s (str): new gene at given index 
        """
        self.genes[index] = s    

    def get_genes(self) -> list:
        return self.genes 
    
    def get_fitness(self) -> int:
        return self.fitness
    
    def update_fitness(self) -> None:
        """update fitness value of a chromosome by counting how many chars are right
        """
        fitness_val = 0
        target_list = list(self.target)
        for i in range(len(self.genes)):
            if self.genes[i] == target_list[i]:
                fitness_val += 1
        self.fitness = fitness_val

    @classmethod
    def generate(cls, parent1, parent2) -> tuple[Self, Self]:
        """generate offsprings of seleceted parents
        0. generete = crossover + mutate
        1. implement the crossover to generate the new generation
        2. mutate the offsprings in the possibility of mutation_rate

        Args:
            parent1 (Chromosome): selected parent1
            parent2 (Chromosome): selected parent2

        Returns:
            tuple[Self, Self]: the generated offsprings
        """
        # generate using crossover
        baby1, baby2 = cls.crossover(parent1, parent2)
        # mutate
        baby1.__mutate()
        baby2.__mutate()
        baby1.update_fitness()
        baby2.update_fitness()
        return baby1, baby2

    @classmethod
    def crossover(cls, parent1: Self, parent2: Self) -> tuple[Self, Self]:
        """implement the crossover to the selected parents
        1. single crossover
        2. randomly select an index in the latter half 
        3. do the crossover in the (index, end) 
        Args:
            parent1 (Chromosome): selected parent1
            parent2 (Chromosome): selected parent2
        Returns:
            tuple[Self, Self]: newborn babies
        """
        p1_genes = parent1.get_genes()
        p2_genes = parent2.get_genes()
        length =cls.length
        cross_index = random.randrange(0, int(length / 3), length)
        baby_genes1 = []
        baby_genes2 = []
        for i in range(0, length):
            if i < 2 * length / 3:
                baby_genes1.append(p1_genes[i])
                baby_genes2.append(p2_genes[i])
            else:
                baby_genes1.append(p2_genes[i])
                baby_genes2.append(p1_genes[i])

        baby1 = Chromosome(baby_genes1)
        baby2 = Chromosome(baby_genes2)
        baby1.update_fitness()
        baby2.update_fitness()
        return baby1, baby2

    @classmethod
    def set_gene_pool(cls, new_gene_pool: str) -> None:
        cls.gene_pool = new_gene_pool
    
    @classmethod
    def set_target(cls, new_target: str) -> None:
        cls.target = new_target
        cls.length = len(new_target)

    @classmethod
    def set_mutation_rate(cls, mutation_rate: float) -> None:
        cls.mutation_rate = mutation_rate
        

class GenericAlgorithm:
    """A class for generating the inputed string using Generic Algorithm
    In the class, when the max_iteration is reached, the process is done
    """
    def __init__(self, size: int = 10, capacity: int = 50, target: str="abc", gene_pool: str="abcdefg", 
                 mutation_rate: float=0.5) -> None:
        """Constructor for Generic Algorithm

        Args:
            size (int, optional): The size of population. Defaults to 10.
            capacity (int, optional): The maximum size of population. Defaults to 50.
            target (str, optional): The target string to be generated. Defaults to "abc".
            gene_pool (str, optional): The pool of genes from which the chromosomes are generated. Defaults to "abcdefg".
            mutation_rate (float, optional): The probability of a gene mutating. Defaults to 0.5.
        """
        self.capacity = capacity
        self.size = size 
        # all the chromosomes are in the list
        self.population = []
        Chromosome.set_mutation_rate(mutation_rate)
        Chromosome.set_target(target)
        Chromosome.set_gene_pool(gene_pool)
        for i in range(self.size):
            chromosome = Chromosome([])
            chromosome.update_fitness()
            self.population.append(chromosome)
        self.iteration = 0
    
    def select_parents(self) -> tuple[Chromosome]:
        """Select the first two individuals with the largest fitness value of Chromosome

        Returns:
            tuple[Chromosome]: The list of reference of selected parents. 
        """
        parent1, parent2 = sorted(self.population, key=lambda chromosome: -chromosome.get_fitness())[:2]
        return parent1, parent2
    
    def die_out(self) -> None:
        """Delete the two individuals with the smallest fitness values
        """
        parent1, parent2 = sorted(self.population, key=lambda chromosome: chromosome.get_fitness())[:2]
        self.population.remove(parent1)
        self.population.remove(parent2)
    
    def run(self, max_iteration = 1000) -> None:
        """To run the procedure of the Generic Algorithm

        Args:
            max_iteration (int, optional): The maximum times of iteration. Defaults to 1000.
        """
        for i in range(max_iteration):
            parent1, parent2 = self.select_parents()
            baby1, baby2 = Chromosome.generate(parent1, parent2)
            self.population.append(baby1)
            self.population.append(baby2)
            self.die_out()
            # the best is whom with the highest fitness score
            best_individual = sorted(self.population, key=lambda chromosome: -chromosome.get_fitness())[0]

            # print the the times of iteration when the best individual is found
            print("The best individual is \"{}\", whose fitness value is {}".format(best_individual.to_string(), best_individual.get_fitness()))
            if best_individual.to_string() == Chromosome.target:
                print("After {} iterations, inputed strings have been generated".format(i))
                break

def main():
    mutation_rate = 0.1
    target = "An idiot with a plan can beat a genius with a plan."
    gene_pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.!? "
    ga = GenericAlgorithm(size=20, capacity=50, target=target, gene_pool=gene_pool, mutation_rate=mutation_rate)
    ga.run(100000)


if __name__ == "__main__":
    main()

    
