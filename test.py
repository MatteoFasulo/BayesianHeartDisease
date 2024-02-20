import random

lista_numeri = []
for i in range(10):
    numero = random.randint(-50, 51)
    lista_numeri.append(numero)

print(lista_numeri)

massimo = max(lista_numeri)
if massimo % 2 == 0:
    print(f"Il massimo numero pari è {massimo}")

lista_numeri.sort()
print(lista_numeri)
