import random

with open("input.txt", "r") as file:
    population, total_run = file.readline().strip().split(" ")
    population, total_run = int(population), int(total_run)
    scoreboard, player_list = {}, []
    for i in range(int(population)):
        name, run = file.readline().strip().split(" ")
        scoreboard[name] = int(run)
        player_list.append(name)

def chromosome(player_list):
    select_player, selection = [], []
    for i in range(population):
        name = random.choice(player_list)
        select_player.append(name)
    for i in range (population):
        if player_list[i] in select_player: 
            selection.append(1)
        else: selection.append(0)
    return selection

def fitness(selection):
    value = 0   
    for i in range (population): 
        if selection[i] == 1:
            value += scoreboard[list(scoreboard.keys())[i]]
    return value

def crossover(parents_list):
    child_list = []
    for i in range(0,4,2):
        crossover_index = random.randint(0, population-1)
        child_1 = parents_list[i][:crossover_index]+parents_list[i+1][crossover_index:]
        child_2 = parents_list[i+1][:crossover_index]+parents_list[i][crossover_index:]
        child_1, child_2 = mutation(child_1, child_2)
        child_list.append(child_1)
        child_list.append(child_2)
    return child_list
    
def mutation(chromosome_1, chromosome_2):
    chromosome_1[random.randint(0, population)-1] = random.randint(0,1)
    chromosome_2[random.randint(0, population-1)] = random.randint(0,1)
    return chromosome_1, chromosome_2
    
def runChase(player_list):
    def notation(a_list):
        string = ''
        for i in a_list:string += str(i)
        return string
    parents_list = []
    for i in range(4):
        parents_list.append(chromosome(player_list))
    for i in range(1000):
        for parent in parents_list:
            if fitness(parent) == total_run:
                return f'{player_list} \n{notation(parent)}'
        else: parents_list = crossover(parents_list)
    return f'{player_list} \n-1'
        
print(runChase(player_list))