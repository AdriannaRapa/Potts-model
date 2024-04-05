""" Wprowadzenie do programowania w języku Python - projekt
    Model Pottsa """

import numpy as np
import random
import matplotlib.pyplot as plt


# Tworzenie siatki losowych stanów
"""Funkcja siatka(q, N) służy do tworzenia siatki losowych stanów spinów o zadanych rozmiarach.
    Opis funkcji siatka(q, N):
     q: liczba wartości, które mogą przyjąć spiny (w modelu Pottsa mogą to być dowolne wartości)
     N: rozmiar siatki (liczba wierszy i kolumn)

     Kroki funkcji siatka(q, N):
     1. Wykorzystuje bibliotekę NumPy i funkcję np.random.randint() do generowania macierzy losowych 
     liczb całkowitych o rozmiarze (N, N).
     2. Zakres losowanych liczb wynosi od 0 do q-1, co odpowiada możliwym wartościom spinów w siatce.
     3. Wygenerowana macierz jest zwracana jako wynik funkcji."""

def siatka(q, N):
    spins = np.random.randint(0, q, size=(N, N))
    return spins


# Funkcja obliczająca zmianę energii przy zmianie pojedynczego spinu
""" Funkcja zmiana_energii(spins, i, j, J, N) służy do obliczania zmiany energii przy zmianie pojedynczego 
spinu na pozycji (i, j) w siatce.
Opis funkcji zmiana_energii(spins, i, j, J, N):
spins: siatka stanów spinów
i: indeks wiersza spinu, którego zmiana energii jest obliczana
j: indeks kolumny spinu, którego zmiana energii jest obliczana
J: stała oddziaływania
N: liczba wierszy i kolumn w siatce

Kroki funkcji zmiana_energii(spins, i, j, J, N):
1. Na podstawie siatki spins i indeksów (i, j) są obliczane indeksy sąsiadujących spinów. Wykorzystuje się 
modulo (%) w celu zapewnienia periodycznych warunków brzegowych, co oznacza, że spin na krańcu siatki ma sąsiadów 
na przeciwnym krańcu.
2. Sumowane są wartości spinów sąsiadów spinu (i, j) i wynik jest przypisywany do zmiennej s.
3. Obliczana jest zmiana energii (dE) zgodnie z równaniem: dE = -J * (s * spins[i, j]). W tym równaniu J 
oznacza stałą oddziaływania, a spins[i, j] to wartość spinu na pozycji (i, j) w siatce.
4. Zmiana energii dE jest zwracana jako wynik funkcji."""

def zmiana_energii(spins, i, j, J, N):
    s = spins[(i + 1) % N, j] + spins[i - 1, j] + spins[i, (j + 1) % N] + spins[i, j - 1]
    dE = -J * (s * spins[i, j])
    return dE


# Wyliczenie prawdopodobieństwa akceptacji zmiany stanu
""" Funkcja prawdpodobienstwo(dE, k, T) służy do obliczania prawdopodobieństwa 
akceptacji zmiany stanu spinu na podstawie zmiany energii (dE), temperatury (T) i stałej Boltzmanna (k).
Opis funkcji prawdpodobienstwo(dE, k, T):
dE: zmiana energii spowodowana zmianą stanu spinu
k: stała Boltzmanna
T: temperatura

Kroki funkcji prawdpodobienstwo(dE, k, T):
1. Obliczane jest prawdopodobieństwo akceptacji zmiany stanu na podstawie wzoru: p = np.exp(-dE / (k * T)). 
Wykorzystuje się funkcję eksponencjalną (np.exp) z modułu NumPy, aby obliczyć wykładnik o wartości -dE / (k * T).
2. Obliczone prawdopodobieństwo akceptacji p jest zwracane jako wynik funkcji.
"""

def prawdpodobienstwo(dE, k, T):
    p = np.exp(-dE / (k * T))
    return p


# Wyliczenie wartości Hamiltonianu dla całego układu
""" Funkcja "hamiltonian" służy do obliczenia wartości Hamiltonianu, czyli energii całej siatki spins 
Opis funkcji hamiltonian(spins):
spins: siatka spinów przyjmujących q różnych wartości

Kroki funckji hamiltonian(spins):
1. Wyzerowanie zmiennej energy
2. Przejście przez wszystkie elementy siatki i obliczenie sumy wartości spinów będących sąsiadami spinu [i,j]
3. Obliczenie energii układu dla spinu [i,j] zgodnie ze wzorem na Hamiltonian energy += -J * s * nb_sum oraz
oraz dodanie wyniku do zmiennej energy
4. Wynikiem funkcji jest wartość zmiennej energy po przejściu pętli przez całą macierz """

def hamiltonian(spins):
    energy = 0
    for i in range(N):
        for j in range(N):
            s = spins[i, j]
            nb_sum = spins[(i + 1) % N, j] + spins[i - 1, j] + spins[i, (j + 1) % N] + spins[i, j - 1]
            energy += -J * s * nb_sum
    return energy


""" Funkcja MonteCarlo(nsteps, k, T, spins) implementuje algorytm Monte Carlo, który wykonuje symulację procesu 
termodynamicznego na siatce spinów.
Opis funkcji MonteCarlo(nsteps, k, T, spins):
nsteps: liczba kroków Monte Carlo, czyli liczba iteracji, które zostaną wykonane w symulacji.
k: stała Boltzmanna.
T: temperatura.
spins: początkowy stan siatki spinów.
q: liczba różnych stanów, jakie mogą przyjąć spiny.

Kroki funkcji MonteCarlo(nsteps, k, T, spins):
1. Utworzenie pustych list state_counts (służy do przechowywania informacji o liczbie różnych stanów wystepujących
w siatce dla kolejnych kroków symulacji), energies (służy do przechowywania informacji o wartościach Hamiltonianu 
dla kolejnych kroków symulacji), state_occurrences (służy do przechowywania informacji o liczbie wystąpień poszczególnych stanów
w siatce dla kolejnych kroków symulacji)
2. Pętla for wykonuje się nsteps razy, czyli tyle, ile określono jako liczbę kroków Monte Carlo.
3. W każdej iteracji losowo wybierane są indeksy i i j, które reprezentują losowy spin w siatce.
4. Obliczana jest zmiana energii (dE) przy zmianie wybranego spinu, wywołując funkcję zmiana_energii(spins, i, j).
5. Wyliczane jest prawdopodobieństwo akceptacji zmiany stanu spinu, wywołując funkcję prawdpodobienstwo(dE, k, T).
5. Sprawdzane jest, czy zmiana zostanie zaakceptowana przez porównanie losowej liczby z prawdopodobieństwem akceptacji. 
Jeśli losowa liczba jest mniejsza od p, to zmiana jest akceptowana.
6. Jeśli zmiana zostanie zaakceptowana, losowany jest nowy stan spinu (new_spin) i aktualizowana jest siatka spins na 
pozycji (i, j) poprzez przypisanie new_spin.
7. Obliczana jest liczba unikatowych wartości przyjmowanych przez spiny i przypisana do zmiennej state_count, 
a następnie dodana do listy state_counts.
8. Aktualizowana jest lista state_occurrences dla każdej wartości q
9. Obliczana jest energia układu z wykorzystaniem funckji hamiltonian(spins) i przypisana do zmiennej energy, 
a następnie dodana do listy energies.
7. Po wykonaniu wszystkich kroków Monte Carlo, zaktualizowana siatka spins jest zwracana jako wynik funkcji.
Ponadto wynikiem funkcji są też listy: state_counts, state_occurrences i energies. """

def MonteCarlo(nsteps, k, T, spins, q):
    state_counts = []
    energies = []
    state_occurrences = {i: [] for i in range(q)}  # Liczba wystąpień poszczególnych stanów
    for step in range(nsteps):
        # Losowe wybieranie spinu
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)

        # Obliczanie zmiany energii
        dE = zmiana_energii(spins, i, j, J, N)

        # Wyliczenie prawdopodobieństwa akceptacji zmiany stanu
        p = prawdpodobienstwo(dE, k, T)

        # Sprawdzenie, czy zmiana zostanie zaakceptowana
        if random.random() < p:
            # Losowe wybranie nowego stanu spinu
            new_spin = random.randint(0, q - 1)
            # Zaktualizowanie siatki
            spins[i, j] = new_spin

        state_count = len(np.unique(spins))
        state_counts.append(state_count)

        # Aktualizacja liczby wystąpień poszczególnych stanów
        for state in range(q):
            state_occurrences[state].append(np.count_nonzero(spins == state))

        energy = hamiltonian(spins)
        energies.append(energy)
    return spins, state_counts, state_occurrences, energies


# Funkcja przedstawiająca wizualizację siatki spinów
"""Funkcja wizualizacja(spins) służy do wizualizacji siatki spinów przy użyciu biblioteki matplotlib.
Opis funkcji wizualizacja(spins):
spins: siatka spinów, która ma zostać zwizualizowana.

Kroki funkcji wizualizacja(spins):
1. Wywołanie funkcji imshow(spins, cmap='hot', interpolation='nearest') z biblioteki matplotlib.pyplot. 
Ta funkcja służy do wyświetlania obrazów i przyjmuje jako argumenty siatkę spins oraz opcje konfiguracyjne. 
cmap='hot' określa paletę kolorów, która będzie użyta do wizualizacji, a interpolation='nearest' ustala metodę 
interpolacji pikseli, aby obraz był wyświetlany bez rozmycia.
2. Wywołanie funkcji colorbar() z biblioteki matplotlib.pyplot. Ta funkcja dodaje kolorową skalę (legendę) do wykresu, 
która informuje o przyporządkowaniu kolorów do wartości na siatce spinów.
3. Wywołanie funkcji title("Siatka stanów") z biblioteki matplotlib.pyplot. Ta funkcja ustawia tytuł wykresu na 
"Siatka stanów".
4. Wywołanie funkcji show() z biblioteki matplotlib.pyplot. Ta funkcja wyświetla wykres siatki spinów."""

def wizualizacja(spins):
    plt.imshow(spins, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Siatka stanów")
    plt.show()


# Funkcja przedstawiająca wykres liczebności stanów w kolejnych krokach symulacji
"""Funkcja wykres_liczebnosci(nsteps, state_counts) służy do przedstawienia wykresu zależności liczby różnych stanów 
od kroku symulacji przy użyciu biblioteki matplotlib.
Opis funkcji wykres_liczebnosci(nsteps, state_counts):
nsteps: liczba kroków symulacji
state_counts: lista przechowująca informacje o liczbie wartości spinów wystepujących w siatce spins w każdym kroku symulacji

Kroki funkcji wizualizacja(spins):
1. Wywołanie funkcji plot(range(nsteps), state_counts)z biblioteki matplotlib.pyplot. 
Ta funkcja służy do utworzenia wykresu, gdzie w zakresie nsteps dla każdego kroku przypisana jest wartość odpowiadająca liczbie
unikatowych stanów wystepujących w siatce w danym kroku symulacji.
2. Wywołanie funkcji xlabel("Krok symulacji") i ylabel("Liczba różnych stanów") z biblioteki matplotlib.pyplot. 
Ta funkcje dodaje odpowiadają za dodanie odpowiednich etykiet dla osi X i dla osi Y.
3. Wywołanie funkcji title("Zależność liczby różnych stanów od kroku symulacji") z biblioteki matplotlib.pyplot. 
Ta funkcja ustawia tytuł wykresu na "Zależność liczby różnych stanów od kroku symulacji".
4. Wywołanie funkcji show() z biblioteki matplotlib.pyplot. Ta funkcja wyświetla wykres zależności liczby różnych stanów 
od kroku symulacji."""

def wykres_liczebnosci(nsteps, state_counts):
    plt.plot(range(nsteps), state_counts)
    plt.xlabel("Krok symulacji")
    plt.ylabel("Liczba różnych stanów")
    plt.title("Zależność liczby różnych stanów od kroku symulacji")
    plt.show()


# Funkcja przedstawiająca wykres liczebności poszczególnych stanów w kolejnych krokach symulacji
"""Funkcja wykres_stanow(q, nsteps, state_occurrences) służy do przedstawienia wykresu zależności liczby wystapień poszczególnych
stanów od kroku symulacji przy użyciu biblioteki matplotlib.
Opis funkcji wykres_stanow(q, nsteps, state_occurrences):
q: liczba stanów, które mogą przyjąć spiny
nsteps: liczba kroków symulacji
state_occurrences: lista przechowująca informacje o liczbie wystąpień poszczególnych stanóww siatce dla kolejnych kroków symulacji

Kroki funkcji wykres_stanow(q, nsteps, state_occurrences):
1. Wywołanie funkcji figure(figsize=(8, 6)) z biblioteki matplotlib.pyplot. 
Ta funkcja służy do utworzenia wykresu o wymiarach 8x6 jednostek.
2. W kolejnym kroku następuje iterowanie po kolejnych stanach spinów od 0 do q-1, gdzie w każdym kroku tworzona jest linia 
przedstawiająca zależność liczby wystąpień danego stanu (state_occurrences[state]) od kroku symulacji (nsteps).
Dodatkowo dla każdego stanu przypisana jest etykieta informująca o numerze prezentowanego stanu.
3. Wywołanie funkcji xlabel("Krok symulacji") i ylabel("Liczba wystąpień") z biblioteki matplotlib.pyplot. 
Ta funkcje dodaje odpowiadają za dodanie odpowiednich etykiet dla osi X i dla osi Y.
4. Wywołanie funkcji title("Zależność liczby wystąpień poszczególnych stanów od kroku symulacji") z biblioteki matplotlib.pyplot. 
Ta funkcja ustawia tytuł wykresu na "Zależność liczby wystąpień poszczególnych stanów od kroku symulacji".
5. Wywołanie funkcji legend() z biblioteki matplotlib.pyplot. Ta funckja wyświetla legendę przedstawiającą dopasowanie kolorów 
linii na wykresie do odpowiednich stanów.
6. Wywołanie funkcji show() z biblioteki matplotlib.pyplot. Ta funkcja wyświetla wykres zależności liczby wystapień poszczególnych
stanów od kroku symulacji."""

def wykres_stanow(q, nsteps, state_occurrences):
    plt.figure(figsize=(8, 6))
    for state in range(q):
        plt.plot(range(nsteps), state_occurrences[state], label=f"Stan {state}")
    plt.xlabel("Krok symulacji")
    plt.ylabel("Liczba wystąpień")
    plt.title("Zależność liczby wystąpień poszczególnych stanów od kroku symulacji")
    plt.legend()
    plt.show()


# Funkcja przedstawiająca wykres liczebności stanów w kolejnych krokach symulacji
"""Funkcja wykres_hamiltonian(nsteps, energies) służy do przedstawienia wykresu zależności wartości Hamiltonianu
od kroku symulacji przy użyciu biblioteki matplotlib.
Opis funkcji wykres_hamiltonian(nsteps, energies):
nsteps: liczba kroków symulacji
energies: lista przechowująca informacje o wartościach Hamiltonianu w każdym kroku symulacji

Kroki funkcji wykres_hamiltonian(nsteps, energies):
1. Wywołanie funkcji plot(range(nsteps), energies) z biblioteki matplotlib.pyplot. 
Ta funkcja służy do utworzenia wykresu, gdzie w zakresie nsteps dla każdego kroku przypisana jest wartość odpowiadająca wartości 
Hamiltonianu w danym kroku symulacji.
2. Wywołanie funkcji xlabel("Krok symulacji") i ylabel("Hamiltonian") z biblioteki matplotlib.pyplot. 
Ta funkcje dodaje odpowiadają za dodanie odpowiednich etykiet dla osi X i dla osi Y.
3. Wywołanie funkcji title("Wartość Hamiltonianu w kolejnych krokach symulacji") z biblioteki matplotlib.pyplot. 
Ta funkcja ustawia tytuł wykresu na "Wartość Hamiltonianu w kolejnych krokach symulacji".
4. Wywołanie funkcji show() z biblioteki matplotlib.pyplot. Ta funkcja wyświetla wykres zależności wartości Hamiltonianu
od kroku symulacji."""
def wykres_hamiltonian(nsteps, energies):
    plt.plot(range(nsteps), energies)
    plt.xlabel("Krok symulacji")
    plt.ylabel("Hamiltonian")
    plt.title("Wartość Hamiltonianu w kolejnych krokach symulacji")
    plt.show()


# Rozmiar siatki
"""Użytkownik jest proszony o wprowadzenie rozmiaru siatki spinów. 
Wartość wprowadzana jest jako liczba całkowita i przypisywana do zmiennej N"""

print(
    "Rozmiar sieci określa utworzenie macierzy, na przykład, "
    "wprowadzenie N = 10 spowoduje utworzenie sieci o wymiarach 10x10")
N = int(input("Podaj rozmiar siatki: "))


# Liczba wartości, które mogą przyjąć spiny
"""Użytkownik jest proszony o wprowadzenie liczby wartości, które mogą być przyjmowane przez spiny. 
Wartość wprowadzana jest jako liczba całkowita i przypisywana do zmiennej q"""

print(
    "Liczba wartości, które mogą być przyjmowane przez spiny określa różnorodność stanów spinów. "
    "Na przykład, wprowadzenie q = 2 oznacza, że spiny mogą przyjąć tylko 2 różne wartości.")
q = int(input("Liczba wartości, które mogą przyjąć spiny: "))


# Temperatura
"""Użytkownik jest proszony o wprowadzenie temperatury symulacji. Wartość 
wprowadzana jest jako liczba zmiennoprzecinkowa i przypisywana do zmiennej T"""

print("Temperatura symulacji określa jej intensywność."
      "Wprowadź wartość temperatury, na przykład 2.0.")
T = float(input("Temperatura: "))


# Stała Boltzmana
""" Stała Boltzmanna (k) jest zdefiniowana jako wartość liczby zmiennoprzecinkowej. 
Jest używana do obliczenia prawdopodobieństwa akceptacji zmiany stanu spinu w symulacji Monte Carlo."""

k = 1.38 * 10 ** -23


# Stała oddziaływania
"""Użytkownik jest proszony o wprowadzenie wartości stałej oddziaływania. 
Wartość wprowadzana jest jako liczba zmiennoprzecinkowa i przypisywana do zmiennej J."""

print("Stała oddziaływania określa siłę oddziaływania między sąsiednimi spinami."
      "Im większa wartość stałej oddziaływania, tym większe znaczenie ma wzajemne wpływanie spinów na siebie.")
J = float(input("Stała oddziaływania: "))


# Liczba kroków MC
"""Użytkownik jest proszony o wprowadzenie liczby kroków Monte Carlo (MC). 
Wartość wprowadzana jest jako liczba całkowita i przypisywana do zmiennej nsteps."""

print("Liczba kroków MC określa liczbę iteracji w metodzie Monte Carlo."
      "Większa liczba kroków MC może prowadzić do dokładniejszych wyników, ale zwiększa czas obliczeń.")
nsteps = int(input("Liczba kroków MC: "))


"""Wywołanie funkcji siatka(q, N), która tworzy siatkę losowych stanów spinów na podstawie wcześniej wprowadzonych 
wartości q i N. 
Wynik tej funkcji przypisywany jest do zmiennej spins"""

print("Generowanie siatki losowych stanów spinów...")
spins = siatka(q, N)
print("Siatka stanów spinów została wygenerowana.")


""" Wywołanie funkcji hamiltonian(spins). Obliczenie wartośći Hamiltonianu układu przed rozpoczęciem symulacji. """

H = hamiltonian(spins)
print("Wartość Hamiltonianu przed wykonaniem symulacji:", H)


""" Wywołanie funkcji wizualizacja(spins). Wyświetlenie siatki spinów przed rozpoczęciem symulacji."""

wizualizacja(spins)


"""Wywołanie funkcji MonteCarlo(nsteps, k, T, spins), która przeprowadza symulację 
Monte Carlo dla określonej liczby kroków nsteps i podanych wartości k, T oraz spins."""

print("Przeprowadzanie symulacji Monte Carlo")
spins, state_counts, state_occurrences, energies = MonteCarlo(nsteps, k, T, spins, q)
print("Symulacja Monte Carlo została zakończona.")


# Wyświetlenie wyniku
"""Wyświetlenie wyniku symulacji, czyli aktualnego stanu siatki spinów."""

print("Siatka spinów po przeprowadzeniu symulacji:")
print(spins)


""" Wywołanie funkcji wizualizacja(spins). Wyświetlenie siatki spinów po zakończeniu symulacji."""

wizualizacja(spins)


""" Wywołanie funkcji hamiltonian(spins). Obliczenie wartośći Hamiltonianu układu po zakończeniu symulacji. """

H = hamiltonian(spins)
print("Wartość Hamiltonianu po wykonaniu symulacji:", H)


""" Wywołanie funkcji wykres_liczebnosci(nsteps, state_counts). Wyświetlenie wykresu zależności liczby wystapień poszczególnych
stanów od kroku symulacji. """

wykres_liczebnosci(nsteps, state_counts)


""" Wywołanie funkcji wykres_stanow(q, nsteps, state_occurrences). Wyświetlenie wykresu zależności liczby wystapień poszczególnych
stanów od kroku symulacji. """

wykres_stanow(q, nsteps, state_occurrences)


""" Wywołanie funkcji wykres_hamiltonian(nsteps, energies). Wyświetlenie wykresu zależności wartości Hamiltonianu
od kroku symulacji. """

wykres_hamiltonian(nsteps, energies)
