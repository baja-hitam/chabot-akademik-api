from pydantic import BaseModel

class ProdiBase(BaseModel):
    nama_prodi: str

class ProdiCreate(ProdiBase):
    pass

class ProdiResponse(ProdiBase):
    kd_prodi: int
    
    class Config:
        from_attributes = True

class ProdiResponseApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: list[ProdiResponse]

class ProdiDetailResponseApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: ProdiResponse

class ProdiUtils(BaseModel):
    value: int
    label: str

class ProdiResponseUtilsApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: list[ProdiUtils]

class ProdiResponseCreateApi(BaseModel):
    responseStatus: bool
    responseMessage: str
    responseBody: None
