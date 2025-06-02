from flask_wtf import FlaskForm
from marshmallow import Schema, fields
from wtforms import StringField
from wtforms.validators import DataRequired


class AuthorizeReq(FlaskForm):
    code = StringField("code",
                       validators=[DataRequired("code代码不能为空")])


class AuthorizeResp(Schema):
    """第三方授权认证响应结构"""
    access_token = fields.String()
    expire_at = fields.Integer()
