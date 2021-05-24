{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Fertility_A01351831.ipynb",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Rulasmaster/A01351831/blob/Report/Fertility_A01351831.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZUnh7phX77zt",
        "outputId": "df0bcda8-fb8a-4260-b08c-df4a766341cf"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from google.colab import drive\n",
        "\n",
        "drive.mount(\"/content/gdrive\")  \n",
        "!pwd"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Mounted at /content/gdrive\n",
            "/content\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sBbW_P1U8XCN",
        "outputId": "a36e7905-b53f-419b-a08f-65f557c71a56"
      },
      "source": [
        "%cd \"/content/gdrive/My Drive/Proyecto Sistemas\"\n",
        "!ls"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/content/gdrive/My Drive/Proyecto Sistemas\n",
            " Fertility_A01351831.ipynb   fertility_Diagnosis.txt\n",
            "'Fertility Diagnosis.docx'\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "D5QmmMLY_sDR",
        "outputId": "8456da57-c00e-4b80-926b-0938d44c8bc1"
      },
      "source": [
        "df = pd.read_csv('fertility_Diagnosis.txt')\n",
        "print(df.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   -0.33  0.69  0  1  1.1  0.1  0.8  0.2  0.88  N\n",
            "0  -0.33  0.94  1  0    1    0  0.8    1  0.31  O\n",
            "1  -0.33  0.50  1  0    0    0  1.0   -1  0.50  N\n",
            "2  -0.33  0.75  0  1    1    0  1.0   -1  0.38  N\n",
            "3  -0.33  0.67  1  1    0    0  0.8   -1  0.50  O\n",
            "4  -0.33  0.67  1  0    1    0  0.8    0  0.50  N\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pSrVemoV9yhG",
        "outputId": "8e4afbd3-95e8-427c-e5e1-8646a984e1b7"
      },
      "source": [
        "columns=[\"Season\", \"Age\", \"Childish_desease\", \"Accident_or_serious_trauma\", \"Surgical_intervention\", \"High_fevers_in_past_years\", \"Alcohol_consumtion\", \"Smoking_Habit\", \"Sitting_hours\", \"Output\"]\n",
        "df = pd.read_csv('fertility_Diagnosis.txt',names=columns)\n",
        "print(df.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   Season   Age  Childish_desease  ...  Smoking_Habit  Sitting_hours  Output\n",
            "0   -0.33  0.69                 0  ...              0           0.88       N\n",
            "1   -0.33  0.94                 1  ...              1           0.31       O\n",
            "2   -0.33  0.50                 1  ...             -1           0.50       N\n",
            "3   -0.33  0.75                 0  ...             -1           0.38       N\n",
            "4   -0.33  0.67                 1  ...             -1           0.50       O\n",
            "\n",
            "[5 rows x 10 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "i4AM2avt_9cx",
        "outputId": "f9ca077b-d149-40af-ebee-f4aabde01a56"
      },
      "source": [
        "df_y=df[['Output']]\n",
        "print(df_y.head())\n",
        "df_x=df[['Season', 'Age', 'Childish_desease', 'Accident_or_serious_trauma', 'Surgical_intervention', 'High_fevers_in_past_years', 'Alcohol_consumtion', 'Smoking_Habit', 'Sitting_hours']]\n",
        "print(df_x.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "  Output\n",
            "0      N\n",
            "1      O\n",
            "2      N\n",
            "3      N\n",
            "4      O\n",
            "   Season   Age  ...  Smoking_Habit  Sitting_hours\n",
            "0   -0.33  0.69  ...              0           0.88\n",
            "1   -0.33  0.94  ...              1           0.31\n",
            "2   -0.33  0.50  ...             -1           0.50\n",
            "3   -0.33  0.75  ...             -1           0.38\n",
            "4   -0.33  0.67  ...             -1           0.50\n",
            "\n",
            "[5 rows x 9 columns]\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Dqkafl-NjunR",
        "outputId": "f8d3f5e9-72b7-430e-caba-585d923e3b12"
      },
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "dfx_train,dfx_test,dfy_train,dfy_test=train_test_split(df_x,df_y,test_size=0.2)\n",
        "print(\"\"\"\n",
        "The training data\n",
        "\"\"\")\n",
        "print(dfx_train.head())\n",
        "print(dfy_train.head())\n",
        "print(\"\"\"\n",
        "The test data\n",
        "\"\"\")\n",
        "print(dfx_test.head())\n",
        "print(dfy_test.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "The training data\n",
            "\n",
            "    Season   Age  ...  Smoking_Habit  Sitting_hours\n",
            "80   -0.33  0.92  ...             -1           0.63\n",
            "6    -0.33  0.67  ...             -1           0.44\n",
            "5    -0.33  0.67  ...              0           0.50\n",
            "47   -0.33  0.72  ...              1           0.19\n",
            "16    1.00  0.64  ...             -1           0.38\n",
            "\n",
            "[5 rows x 9 columns]\n",
            "   Output\n",
            "80      N\n",
            "6       N\n",
            "5       N\n",
            "47      N\n",
            "16      N\n",
            "\n",
            "The test data\n",
            "\n",
            "    Season   Age  ...  Smoking_Habit  Sitting_hours\n",
            "7    -0.33  1.00  ...             -1           0.38\n",
            "12    1.00  0.75  ...              1           0.25\n",
            "64   -1.00  0.53  ...             -1           0.31\n",
            "88   -0.33  0.83  ...             -1           0.31\n",
            "92    0.33  0.75  ...             -1           0.38\n",
            "\n",
            "[5 rows x 9 columns]\n",
            "   Output\n",
            "7       N\n",
            "12      N\n",
            "64      N\n",
            "88      N\n",
            "92      N\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pqVFLARNBv64",
        "outputId": "92450f0a-6440-41f5-eeda-7718c8f5a91a"
      },
      "source": [
        "from sklearn.tree import DecisionTreeClassifier\n",
        "t_classif = DecisionTreeClassifier(max_depth = 6)\n",
        "t_classif.fit(dfx_train,dfy_train)\n",
        "\n",
        "print(\"Tree Classifier Configuration\")\n",
        "print (t_classif)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Tree Classifier Configuration\n",
            "DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',\n",
            "                       max_depth=6, max_features=None, max_leaf_nodes=None,\n",
            "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
            "                       min_samples_leaf=1, min_samples_split=2,\n",
            "                       min_weight_fraction_leaf=0.0, presort='deprecated',\n",
            "                       random_state=None, splitter='best')\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AIoayZZLDKxL",
        "outputId": "f281755b-3545-4627-a69d-1f77200aa893"
      },
      "source": [
        "print(\"\"\"\n",
        "Test data results\n",
        "\"\"\")\n",
        "Test_results = pd.DataFrame(t_classif.predict(dfx_test))\n",
        "print(Test_results)\n",
        "print(\"\"\"\n",
        "Original data results\n",
        "\"\"\")\n",
        "print(dfy_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\n",
            "Test data results\n",
            "\n",
            "    0\n",
            "0   N\n",
            "1   N\n",
            "2   N\n",
            "3   N\n",
            "4   O\n",
            "5   N\n",
            "6   N\n",
            "7   N\n",
            "8   O\n",
            "9   N\n",
            "10  N\n",
            "11  N\n",
            "12  N\n",
            "13  N\n",
            "14  N\n",
            "15  N\n",
            "16  N\n",
            "17  O\n",
            "18  N\n",
            "19  N\n",
            "\n",
            "Original data results\n",
            "\n",
            "   Output\n",
            "7       N\n",
            "12      N\n",
            "64      N\n",
            "88      N\n",
            "92      N\n",
            "82      N\n",
            "48      N\n",
            "55      N\n",
            "23      O\n",
            "77      N\n",
            "42      N\n",
            "81      N\n",
            "9       N\n",
            "26      O\n",
            "85      N\n",
            "87      N\n",
            "52      N\n",
            "91      N\n",
            "39      N\n",
            "11      N\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nBPRegg2WGOr",
        "outputId": "474dfe08-b38e-4b39-9855-3e5eb6b6ee83"
      },
      "source": [
        "from sklearn import preprocessing\n",
        "label_Test = preprocessing.LabelEncoder()\n",
        "Rtest = dfy_test.apply(label_Test.fit_transform)\n",
        "print(Rtest.head())\n",
        "print(len(Rtest))\n",
        "\n",
        "Rresults = Test_results.apply(label_Test.fit_transform)\n",
        "print(Rresults.head())\n",
        "print(len(Rresults))\n",
        "\n",
        "print(\"\")\n",
        "from sklearn.metrics import accuracy_score\n",
        "print(\"Accuracy:\",accuracy_score(Rresults,Rtest))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "    Output\n",
            "7        0\n",
            "12       0\n",
            "64       0\n",
            "88       0\n",
            "92       0\n",
            "20\n",
            "   0\n",
            "0  0\n",
            "1  0\n",
            "2  0\n",
            "3  0\n",
            "4  1\n",
            "20\n",
            "\n",
            "Accuracy: 0.85\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 416
        },
        "id": "vW8xzBabeGNf",
        "outputId": "4a3e3e40-6c4d-4222-b083-4ff66996a715"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "X = ['N','O']\n",
        "Val1=Test_results.value_counts()\n",
        "Val2=dfy_test.value_counts()\n",
        "print(Val1)\n",
        "print(Val2)\n",
        "x_axis=np.arange(len(X))\n",
        "print(x_axis)\n",
        "\n",
        "plt.bar(x_axis - 0.2, Val1,0.4, label=\"N\")\n",
        "plt.bar(x_axis + 0.2, Val2,0.4, label=\"O\")"
      ],
      "execution_count": 86,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "N    17\n",
            "O     3\n",
            "dtype: int64\n",
            "Output\n",
            "N         18\n",
            "O          2\n",
            "dtype: int64\n",
            "[0 1]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<BarContainer object of 2 artists>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 86
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PvmBsP8D6dIq",
        "outputId": "b3ac2cc1-829c-42dc-f7a0-e79c6ada0ac9"
      },
      "source": [
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "rnd_clf = RandomForestClassifier(n_estimators=3, n_jobs=-1, random_state=42)\n",
        "rnd_clf.fit(dfx_train,dfy_train)\n",
        "Rfpred= pd.DataFrame(rnd_clf.predict(dfx_test))\n",
        "\n",
        "print(rnd_clf)\n",
        "print(Rfpred)\n",
        "print(\"random forest\", accuracy_score(dfy_test, Rfpred))"
      ],
      "execution_count": 90,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().\n",
            "  after removing the cwd from sys.path.\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
            "                       criterion='gini', max_depth=None, max_features='auto',\n",
            "                       max_leaf_nodes=None, max_samples=None,\n",
            "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
            "                       min_samples_leaf=1, min_samples_split=2,\n",
            "                       min_weight_fraction_leaf=0.0, n_estimators=3, n_jobs=-1,\n",
            "                       oob_score=False, random_state=42, verbose=0,\n",
            "                       warm_start=False)\n",
            "    0\n",
            "0   N\n",
            "1   N\n",
            "2   N\n",
            "3   N\n",
            "4   N\n",
            "5   N\n",
            "6   N\n",
            "7   N\n",
            "8   N\n",
            "9   N\n",
            "10  N\n",
            "11  N\n",
            "12  N\n",
            "13  O\n",
            "14  N\n",
            "15  N\n",
            "16  N\n",
            "17  N\n",
            "18  N\n",
            "19  N\n",
            "random forest 0.95\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 416
        },
        "id": "yRJq-Iv58Mio",
        "outputId": "74dc2473-ef55-4614-c0d8-cac79b877607"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "X = ['N','O']\n",
        "Val3=Rfpred.value_counts()\n",
        "Val4=dfy_test.value_counts()\n",
        "print(Val3)\n",
        "print(Val4)\n",
        "x_axis=np.arange(len(X))\n",
        "print(x_axis)\n",
        "\n",
        "plt.bar(x_axis - 0.2, Val3,0.4, label=\"N\")\n",
        "plt.bar(x_axis + 0.2, Val4,0.4, label=\"O\")"
      ],
      "execution_count": 91,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "N    19\n",
            "O     1\n",
            "dtype: int64\n",
            "Output\n",
            "N         18\n",
            "O          2\n",
            "dtype: int64\n",
            "[0 1]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<BarContainer object of 2 artists>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 91
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OqytIlRLClRp",
        "outputId": "86b77bbf-e74e-4433-a44b-26c71cbaddec"
      },
      "source": [
        "#User Answers\n",
        "Ans=[[None]*9]\n",
        "Nombre = str (input('Whats your name?: '))\n",
        "print(\"\")\n",
        "Ans[0][0] = float (input('Season in which the analysis was performed. Winter(-1) , Spring(-0.33) , Summer(0.33) , Fall(1): '))\n",
        "print(\"\")\n",
        "Edad = int (input('Age at the time of analysis. (18-36): '))\n",
        "Ans[0][1] = (Edad-18)/18\n",
        "print(\"\")\n",
        "Ans[0][2] = int (input('Childish diseases (ie , chicken pox, measles, mumps, polio). Yes(0), No(1): '))\n",
        "print(\"\")\n",
        "Ans[0][3] = int (input('Accident or serious trauma. Yes(0), No(1): '))\n",
        "print(\"\")\n",
        "Ans[0][4] = int (input('Surgical intervention. Yes(0), No(1): '))\n",
        "print(\"\")\n",
        "Ans[0][5] = int (input('High fevers in the last year. Less than three months ago(-1), More than three months ago(0), No(1): '))\n",
        "print(\"\")\n",
        "Ans[0][6] = float (input('Frequency of alcohol consumption. Several times a day(0.2), Every day(0.4), Several times a week(0.6), Once a week(0.8), Hardly ever or never(1): '))\n",
        "print(\"\")\n",
        "Ans[0][7] = int (input('Smoking habit. Never(-1), Occasional(0), Daily(1): '))\n",
        "print(\"\")\n",
        "Horas = int (input('Number of hours spent sitting per day (1-16): '))\n",
        "Ans[0][8] = (Horas-1)/15\n",
        "print(\"\")\n",
        "print(\"\")\n",
        "print(Ans)\n",
        "print(\"\")\n",
        "probst = t_classif.predict_proba(Ans)\n",
        "#print(\"probability of class for query\",Ans,probst)\n",
        "print(\"\")\n",
        "predt =  t_classif.predict(Ans)\n",
        "print(\"\"\"\n",
        "According to decision tree\"\"\")\n",
        "print(Nombre,\", your concentration of sperm is: \",predt,\" (normal (N), altered (O))\")\n",
        "\n",
        "probsrf = rnd_clf.predict_proba(Ans)\n",
        "predrf = rnd_clf.predict(Ans) \n",
        "print(\"\"\"\n",
        "According to random forest\"\"\")\n",
        "print(Nombre,\", your concentration of sperm is: \",predrf,\" (normal (N), altered (O))\")"
      ],
      "execution_count": 95,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Whats your name?: Ilse\n",
            "\n",
            "Season in which the analysis was performed. Winter(-1) , Spring(-0.33) , Summer(0.33) , Fall(1): -0.33\n",
            "\n",
            "Age at the time of analysis. (18-36): 36\n",
            "\n",
            "Childish diseases (ie , chicken pox, measles, mumps, polio). Yes(0), No(1): 0\n",
            "\n",
            "Accident or serious trauma. Yes(0), No(1): 0\n",
            "\n",
            "Surgical intervention. Yes(0), No(1): 0\n",
            "\n",
            "High fevers in the last year. Less than three months ago(-1), More than three months ago(0), No(1): 0\n",
            "\n",
            "Frequency of alcohol consumption. Several times a day(0.2), Every day(0.4), Several times a week(0.6), Once a week(0.8), Hardly ever or never(1): 0.6\n",
            "\n",
            "Smoking habit. Never(-1), Occasional(0), Daily(1): 1\n",
            "\n",
            "Number of hours spent sitting per day (1-16): 12\n",
            "\n",
            "\n",
            "[[-0.33, 1.0, 0, 0, 0, 0, 0.6, 1, 0.7333333333333333]]\n",
            "\n",
            "\n",
            "\n",
            "According to decision tree\n",
            "Ilse , your concentration of sperm is:  ['O']  (normal (N), altered (O))\n",
            "\n",
            "According to random forest\n",
            "Ilse , your concentration of sperm is:  ['N']  (normal (N), altered (O))\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}