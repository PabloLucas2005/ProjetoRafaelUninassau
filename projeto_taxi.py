import gymnasium as gym  
import numpy as np  
import matplotlib.pyplot as plt  
import pickle  

def executar(episodios, treinar=True, renderizar=False):
    ambiente = gym.make('Taxi-v3', render_mode='human' if renderizar else None)
    
    matriz_q = np.zeros((ambiente.observation_space.n, ambiente.action_space.n)) if treinar else pickle.load(open('taxi.pkl', 'rb'))
    
    taxa_aprendizado = 0.9  
    fator_desconto = 0.9  
    epsilon = 1.0  
    decaimento_epsilon = 0.0001  
    gerador_aleatorio = np.random.default_rng()  

    recompensas_por_episodio = []  

    for episodio in range(episodios):
        estado, _ = ambiente.reset() 
        terminado, truncado = False, False  
        recompensas_totais = 0 

        while not (terminado or truncado):
          
            acao = ambiente.action_space.sample() if treinar and gerador_aleatorio.random() < epsilon else np.argmax(matriz_q[estado])
            
            novo_estado, recompensa, terminado, truncado, _ = ambiente.step(acao)
            recompensas_totais += recompensa  
            
            if treinar:
                
                matriz_q[estado, acao] += taxa_aprendizado * (recompensa + fator_desconto * np.max(matriz_q[novo_estado]) - matriz_q[estado, acao])
            
            estado = novo_estado
        
       
        epsilon = max(epsilon - decaimento_epsilon, 0)
        
        if epsilon == 0:
            taxa_aprendizado = 0.0001

        recompensas_por_episodio.append(recompensas_totais)  

    ambiente.close()  

  
    recompensas_suavizadas = [np.mean(recompensas_por_episodio[max(0, i-100):i+1]) for i in range(episodios)]
    
    plt.plot(recompensas_suavizadas)
    plt.savefig('taxi.png')  

   
    if treinar:
        pickle.dump(matriz_q, open('taxi.pkl', 'wb'))

if __name__ == '__main__':
    executar(15000)
    
    
    executar(10, treinar=False, renderizar=True)
