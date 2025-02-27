import { useEffect, useState } from "react"
import Error from "../components/Error"
import { getData, handlePromiseState } from "../services/api"
import Input from "../components/Input"
import Form from "../components/Form"
import Loading from "../components/Loading"
import { HexColors } from "../lib/colors"
import PreViewer from "../components/JSONViewer"

export default function InspectView() {
    const [error, setError] = useState<string | null>(null)
    const [id, setId] = useState<string>("")
    const [isValidId, setIsValidId] = useState<boolean>(false)
    const [data, setData] = useState<any>("")
    const [processing, setProcessing] = useState<boolean>(false)

    useEffect(() => {
        if (id && id.length === 24) return setIsValidId(true)
        setIsValidId(false)
    }, [id])

    function handleSubmit(): void {
        // Prevent submission if id is invalid
        if (!isValidId) return
        // Prevent re-submission
        if (processing) return

        setProcessing(true)
        handlePromiseState(getData(id), setData, setError)

        // Reset
        setError(null)
        setProcessing(false)
    }

    function handleId(e: React.ChangeEvent<HTMLInputElement>): void {
        setId(e.target.value)
    }

    return (
        <div className="flex flex-col space-y-2">
            {error && <Error message={error} />}
            <Form onSubmit={handleSubmit} className="flex-row">
                <Input
                    type="text"
                    placeholder="Enter a job ID..."
                    onChange={handleId}
                    className="w-full bg-neutral-500 border-white border-2 hover:bg-neutral-600 focus:bg-neutral-600"
                />
                {(!processing && isValidId) && <Input
                    type="submit"
                    value="Get"
                    className="cursor-pointer"
                />}
                {processing && <Loading color={HexColors.WHITE} />}
            </Form>
            {data && <PreViewer
                value={JSON.stringify(data, null, 2)}
                className="text-white"
            />}
        </div>
    )
}
