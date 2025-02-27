import { useState } from "react"
import JSONViewer from "../components/JSONViewer"
import Error from "../components/Error"
import { getData, handlePromiseState } from "../services/api"
import Input from "../components/Input"
import Label from "../components/Label"
import Form from "../components/Form"

export default function InspectView() {
    const [error, setError] = useState<string | null>(null)
    const [id, setId] = useState<string>("")
    const [data, setData] = useState<Record<string, any> | null>(null)

    function handleSubmit(): void {
        if (!id || id.length === 0) {
            setError("ID is missing")
        }
        handlePromiseState(getData(id), setData, setError)
        setError(null)
    }

    function handleId(e: React.ChangeEvent<HTMLInputElement>): void {
        setId(e.target.value)
    }

    return (
        <div className="flex flex-col space-y-2">
            {error && <Error message={error} />}
            <Form onSubmit={handleSubmit}>
                <Label label="ID" />
                <Input
                    type="text"
                    onChange={handleId}
                    className="w-full bg-neutral-500 border-white border-2 hover:bg-neutral-600"
                />
            </Form>
            {(!error && data) && <JSONViewer
                data={data}
                errorCallback={setError}
                className="text-white"
            />}
        </div>
    )
}
