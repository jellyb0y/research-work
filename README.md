# Репозиторий НИРС
Автор: Кисель Виталий Максимович

# Установка зависимостей и развёртка проекта
- Установка необходимых пакетов: `sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran python3-dev python3-matplotlib`

- Пакетный менеджер `pip install --upgrade pip ezdeps`

- Установка зависимостей `ezdeps install`

- Для numpy вероятно придётся прописать в env `OPENBLAS_CORETYPE=ARMV8`

На всякий случай [дока](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770) по установке TF на плату.
