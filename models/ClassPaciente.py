class Paciente:
    def __init__(self, id, nome, idade, sexo, resultado, criado_em=None):
        self.id = id
        self.nome = nome
        self.idade = idade
        self.sexo = sexo
        self.resultado = resultado
        self.criado_em = criado_em

    def __repr__(self):
        return (
            f"Paciente(id={self.id}, nome='{self.nome}', idade={self.idade}, "
            f"sexo='{self.sexo}', resultado='{self.resultado}', criado_em='{self.criado_em}')"
        )

    def to_dict(self):
        return {
            "id": self.id,
            "nome": self.nome,
            "idade": self.idade,
            "sexo": self.sexo,
            "resultado": self.resultado,
            "criado_em": self.criado_em,
        }
