"""Adapt immediate single-request test doubles to the batch wire contract."""


def batch_methods(submit, get, status):
    batches = {}

    def submit_batch(batch):
        reply = {"accepted": False, "reason": "empty"}
        for request in batch["requests"]:
            reply = submit(request)
            if not reply["accepted"]:
                return reply
        batches[batch["batch_id"]] = batch["requests"]
        return {**reply, "batch_id": batch["batch_id"]}

    def get_batch(value):
        results = [
            {"request_id": request["request_id"], **get(request)}
            for request in batches[value["batch_id"]]
        ]
        first_failure = next(
            (result for result in results if result["status"] != "SUCCEEDED"), None
        )
        return {
            "batch_id": value["batch_id"],
            "epoch": results[0]["epoch"],
            "status": "SUCCEEDED",
            "requests": results,
            **(
                {"status": first_failure["status"], "error": first_failure.get("error")}
                if first_failure
                else {}
            ),
        }

    return {"SubmitBatch": submit_batch, "GetBatch": get_batch, "Status": status}
