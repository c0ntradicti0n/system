import React from "react";
import {Button} from "antd";
import { navigate} from "raviger";

export function ExperimentsView(props) {
    console.log(props)

    return (
<>{props.mods?.map( (mod) =>
    <div><Button onClick={() =>
    {
        navigate(`/editor#hash=${mod.hash}&activeTab=3`
        )
    window.location.reload()

    }}>{mod.hash}</Button></div>)}
    </>

    )
}